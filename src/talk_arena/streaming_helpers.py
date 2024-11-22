import base64
import os
from pathlib import Path

import google.generativeai as genai
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import xxhash
from datasets import Audio
from openai import OpenAI
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    Qwen2AudioForConditionalGeneration,
    TextIteratorStreamer,
    WhisperForConditionalGeneration,
)
from transformers.generation import GenerationConfig


def gradio_gen_factory(streaming_fn, model_name, anonymous):
    def gen_from(audio_input, order):
        resp_gen = streaming_fn(audio_input)
        for resp in resp_gen:
            my_resp = gr.Textbox(
                value=resp,
                visible=True,
                label=model_name if not anonymous else f"Model {order+1}",
            )
            yield my_resp

    return gen_from


def gemini_streaming(model_id):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    resampler = Audio(sampling_rate=16_000)

    model = genai.GenerativeModel(model_id)

    def get_chat_response(audio_input):
        if audio_input == None:
            return ""
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        prompt = "You are a helpful assistant. Respond conversationally to the speech provided."
        inputs = [prompt, {"mime_type": "audio/wav", "data": Path(f"{x}.wav").read_bytes()}]
        text_response = []
        responses = model.generate_content(inputs, stream=True)
        for chunk in responses:
            text_response.append(chunk.text)
            yield "".join(text_response)
        os.remove(f"{x}.wav")
        return "".join(text_response)

    return get_chat_response, model


def geminip_streaming(model_id):
    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/")
    model = client
    resampler = Audio(sampling_rate=16_000)

    def get_chat_response(audio_input):
        if audio_input == None:
            return ""
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.mp3", a["array"], a["sampling_rate"], format="mp3")
        audio_bytes = Path(f"{x}.mp3").read_bytes()
        encoded_data = base64.b64encode(audio_bytes).decode("utf-8")
        prompt = "You are a helpful assistant. Respond conversationally to the speech provided."
        text_response = []
        try:
            response = client.chat.completions.create(
                model="gemini-1.5-pro",
                messages=[
                    {"role": "user", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": "data:audio/mp3;base64,{}".format(encoded_data)}
                        ],
                    },
                ],
            )
            # print('#Response', response.choices[0].message.content)
            os.remove(f"{x}.mp3")
            yield response.choices[0].message.content
            return response.choices[0].message.content
        except:
            return "error"

    return get_chat_response, model


def gpt4o_streaming(model_id):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resampler = Audio(sampling_rate=16_000)

    def get_chat_response(audio_input):
        if audio_input == None:
            return ""
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        with open(f"{x}.wav", "rb") as wav_file:
            wav_data = wav_file.read()
        encoded_string = base64.b64encode(wav_data).decode("utf-8")
        prompt = "You are a helpful assistant. Respond conversationally to the speech provided."
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "input_audio", "input_audio": {"data": encoded_string, "format": "wav"}},
                        ],
                    },
                ],
            )
            os.remove(f"{x}.wav")
            yield completion.choices[0].message.audio.transcript
            return completion.choices[0].message.audio.transcript
        except:
            return "error"

    return get_chat_response, client


def asr_streaming(llm, tokenizer, asr_pipe):
    resampler = Audio(sampling_rate=16_000)

    @torch.no_grad
    def pipelined(audio_input):
        if audio_input == None:
            return ""
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": asr_pipe(f"{x}.wav", generate_kwargs={"task": "transcribe"}, return_timestamps=False)[
                    "text"
                ],
            },
        ]
        inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        streamer = TextIteratorStreamer(tokenizer)
        llm.generate(
            torch.tensor(inputs).unsqueeze(0).to(llm.model.embed_tokens.weight.device),
            streamer=streamer,
            max_length=256,
        )
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace(
                "<|eot_id|>", ""
            )
        return generated_text

    os.remove(f"{x}.wav")
    return pipelined


def diva_streaming(diva_model_str):
    diva_model = AutoModel.from_pretrained(diva_model_str, trust_remote_code=True)
    resampler = Audio(sampling_rate=16_000)

    @torch.no_grad
    def diva_audio(audio_input, do_sample=False, temperature=0.001):
        sr, y = audio_input
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        yield from diva_model.generate_stream(
            a["array"],
            (
                "You are a helpful assistant The user is talking to you with their voice and you are responding with"
                " text."
            ),
            do_sample=do_sample,
            max_new_tokens=256,
        )

    return diva_audio, diva_model


def typhoon_streaming(typhoon_model_str, device="cuda:1"):
    typhoon_model = AutoModel.from_pretrained(typhoon_model_str, trust_remote_code=True).to(device)
    tokenizer = typhoon_model.llama_tokenizer
    resampler = Audio(sampling_rate=16_000)

    @torch.no_grad
    def typhoon_audio(audio_input, do_sample=False, temperature=0.001):
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        streamer = TextIteratorStreamer(tokenizer)
        prompt_pattern = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<Speech><SpeechHere></Speech> {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        response = typhoon_model.generate(
            wav_path=f"{x}.wav",
            prompt=(
                "You are a helpful assistant. Listen to this audio, and respond accordingly in the language it is"
                " spoken in."
            ),
            device=device,
            prompt_pattern=prompt_pattern,
            do_sample=False,
            max_length=1200,
            num_beams=1,
            streamer=streamer,  # supports TextIteratorStreamer
        )
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace(
                "<|eot_id|>", ""
            )
        os.remove(f"{x}.wav")
        return generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "")

    return typhoon_audio, typhoon_model


def qwen2_streaming(qwen2_model_str):
    resampler = Audio(sampling_rate=16_000)
    qwen2_processor = AutoProcessor.from_pretrained(qwen2_model_str)
    qwen2_model = Qwen2AudioForConditionalGeneration.from_pretrained(qwen2_model_str, device_map="auto")
    qwen2_model.generation_config = GenerationConfig.from_pretrained(
        qwen2_model_str,
        trust_remote_code=True,
        do_sample=False,
        top_k=50,
        top_p=1.0,
    )

    @torch.no_grad
    def qwen2_audio(audio_input, do_sample=False, temperature=0.001):
        if audio_input == None:
            return ""
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": f"{x}.wav",
                    },
                ],
            },
        ]
        text = qwen2_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = [librosa.load(f"{x}.wav", sr=qwen2_processor.feature_extractor.sampling_rate)[0]]
        inputs = qwen2_processor(text=text, audios=audios, return_tensors="pt", padding=True)
        streamer = TextIteratorStreamer(qwen2_processor)
        qwen2_model.generate(**inputs, streamer=streamer, max_length=256)
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
        os.remove(f"{x}.wav")
        return generated_text

    return qwen2_audio, qwen2_model
