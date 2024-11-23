import base64
import json
import os
from pathlib import Path

import google.generativeai as genai
import gradio as gr
import numpy as np
import soundfile as sf

import xxhash
from datasets import Audio
from openai import OpenAI

def _get_config_for_model_name(model_id):
    if "API_MODEL_CONFIG" in os.environ:
        return json.loads(os.environ["API_MODEL_CONFIG"])[model_id]
    return {
        "WillHeld/DiVA-llama-3.2-1b": {"base_url": "http://localhost:8002/v1", "api_key": "empty"},
        "scb10x/llama-3-typhoon-v1.5-8b-audio-preview": {"base_url": "http://localhost:8003/v1", "api_key": "empty"},
        "WillHeld/DiVA-llama-3-v0-8b": {"base_url": "http://localhost:8004/v1", "api_key": "empty"},
        "Qwen/Qwen2-Audio-7B-Instruct": {"base_url": "http://localhost:8005/v1", "api_key": "empty"},
    }[model_id]


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


def llm_streaming(model_id: str, prompt: str):
    if "gpt" in model_id:
        client = OpenAI()
    else:
        client = OpenAI(**_get_config_for_model_name(model_id))
    try:
        completion = client.chat.completions.create(
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
        for chunk in completion:
            if len(chunk.choices) > 0:
                text_response.append(chunk.choices[0].delta.content)
                yield "".join(text_response)
        return "".join(text_response)
    except:
        return "error"


def asr_streaming(model_id, asr_pipe):
    resampler = Audio(sampling_rate=16_000)

    def pipelined(audio_input):
        if audio_input == None:
            return ""
        sr, y = audio_input
        x = xxhash.xxh32(bytes(y)).hexdigest()
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        a = resampler.decode_example(resampler.encode_example({"array": y, "sampling_rate": sr}))
        sf.write(f"{x}.wav", a["array"], a["sampling_rate"], format="wav")
        text = asr_pipe(f"{x}.wav", generate_kwargs={"task": "transcribe"}, return_timestamps=False)["text"]
        os.remove(f"{x}.wav")
        return llm_streaming(model_id, prompt=text)

    return pipelined


def api_streaming(model_id):
    client = OpenAI(**_get_config_for_model_name(model_id))
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
            for chunk in completion:
                if len(chunk.choices) > 0:
                    text_response.append(chunk.choices[0].delta.content)
                    yield "".join(text_response)
            os.remove(f"{x}.wav")
            return "".join(text_response)
        except:
            return "error"

    return get_chat_response, client
