import base64
import json
import os
from pathlib import Path
import asyncio

import google.generativeai as genai
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import xxhash
from datasets import Audio
from openai import AsyncOpenAI
from transformers import (
    AutoModel,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TextIteratorStreamer,
)
from transformers.generation import GenerationConfig



def _get_config_for_model_name(model_id):
    if "API_MODEL_CONFIG" in os.environ:
        return json.loads(os.environ["API_MODEL_CONFIG"])[model_id]
    return {
        "pipeline/meta-llama/Meta-Llama-3-8B-Instruct": {"base_url": "http://localhost:8001/v1", "api_key": "empty"},
        "scb10x/llama-3-typhoon-v1.5-8b-audio-preview": {
            "base_url": "http://localhost:8002/v1",
            "api_key": "empty",
        },
        "WillHeld/DiVA-llama-3-v0-8b": {
            "base_url": "http://localhost:8003/v1",
            "api_key": "empty",
        },
        "Qwen/Qwen2-Audio-7B-Instruct": {
            "base_url": "http://localhost:8004/v1",
            "api_key": "empty",
        }
    }[model_id]


def gradio_gen_factory(streaming_fn, model_name, anonymous):
    async def gen_from(audio_input, order):
        with torch.no_grad():
            async for resp in streaming_fn(audio_input):
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

    async def get_chat_response(audio_input):
        if audio_input is None:
            raise StopAsyncIteration("")
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

    return get_chat_response, model


def gpt4o_streaming(model_id):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resampler = Audio(sampling_rate=16_000)

    async def get_chat_response(audio_input):
        if audio_input is None:
            raise StopAsyncIteration("")
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
            completion = await client.chat.completions.create(
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
        except:
            raise StopAsyncIteration("error")

    return get_chat_response, client


async def llm_streaming(model_id: str, prompt: str):
    if "gpt" in model_id:
        client = AsyncOpenAI()
    else:
        client = AsyncOpenAI(**_get_config_for_model_name(model_id))
    try:
        completion = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are helpful assistant."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        text_response = []
        async for chunk in completion:
            if len(chunk.choices) > 0:
                text_response.append(chunk.choices[0].delta.content)
                yield "".join(text_response)
    except:
        raise StopAsyncIteration("error")


def asr_streaming(model_id, asr_pipe):
    resampler = Audio(sampling_rate=16_000)

    async def pipelined(audio_input):
        if audio_input is None:
            raise StopAsyncIteration("")
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        text = await asyncio.to_thread(
            asr_pipe(f"{x}.wav", generate_kwargs={"task": "transcribe"}, return_timestamps=False)["text"]
        )
        os.remove(f"{x}.wav")
        async for response in llm_streaming(model_id, prompt=text):
            yield response

    return pipelined


def api_streaming(model_id):
    client = AsyncOpenAI(**_get_config_for_model_name(model_id))
    resampler = Audio(sampling_rate=16_000)

    async def get_chat_response(audio_input):
        if audio_input is None:
            raise StopAsyncIteration("")
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        with open(f"{x}.wav", "rb") as wav_file:
            wav_data = wav_file.read()
        encoded_string = base64.b64encode(wav_data).decode("utf-8")
        prompt = (
            "You are a helpful assistant. Respond conversationally to the speech provided in the language it is"
            " spoken in."
        )
        try:
            completion = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "audio", "audio_url": "data:audio/wav;base64," + encoded_string},
                        ],
                    },
                ],
                stream=True,
            )
            text_response = []
            async for chunk in completion:
                if len(chunk.choices) > 0:
                    text_response.append(chunk.choices[0].delta.content)
                    yield "".join(text_response)
            os.remove(f"{x}.wav")
        except:
            raise StopAsyncIteration("error")

    return get_chat_response, client


# Local Hosting Utilities


def diva_streaming(diva_model_str):
    diva_model = AutoModel.from_pretrained(diva_model_str, trust_remote_code=True, device_map="balanced_low_0")
    resampler = Audio(sampling_rate=16_000)

    async def diva_audio(audio_input, do_sample=False, temperature=0.001):
        sr, y = audio_input
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        stream = diva_model.generate_stream(
            a["array"],
            (
                "You are a helpful assistant The user is talking to you with their voice and you are responding with"
                " text."
            ),
            do_sample=do_sample,
            max_new_tokens=256,
        )
        for text in stream:
            yield text

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

    async def qwen2_audio(audio_input, do_sample=False, temperature=0.001):
        if audio_input is None:
            raise StopAsyncIteration("")
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
        generation_task = asyncio.create_task(qwen2_model.generate(**inputs, streamer=streamer, max_length=256))

        generated_text = ""
        async for new_text in streamer:
            generated_text += new_text
            yield generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")

        await generation_task
        os.remove(f"{x}.wav")

    return qwen2_audio, qwen2_model

def typhoon_streaming(typhoon_model_str, device="cuda:0"):
    resampler = Audio(sampling_rate=16_000)
    typhoon_model = AutoModel.from_pretrained(typhoon_model_str, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = typhoon_model.llama_tokenizer
    typhoon_model.to(device)
    typhoon_model.eval()
    prompt_pattern = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<Speech><SpeechHere></Speech>"
        " {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    prompt = (
        "You are a helpful assistant. Respond conversationally to the speech provided in the language it is spoken in."
    )

    async def typhoon_audio(audio_input, do_sample=False, temperature=0.001):
        if audio_input == None:
            raise StopAsyncIteration("")
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        streamer = TextIteratorStreamer(tokenizer)
        generation_task = asyncio.create_task(
            typhoon_model.generate(
                audio=a["array"],
                prompt=prompt,
                prompt_pattern=prompt_pattern,
                device=device,
                do_sample=False,
                max_length=1200,
                num_beams=1,
                streamer=streamer,  # supports TextIteratorStreamer
            )
        )
        generated_text = ""
        async for new_text in streamer:
            generated_text += new_text
            yield generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace(
                "<|eot_id|>", ""
            )
        await generation_task

    return typhoon_audio, typhoon_model