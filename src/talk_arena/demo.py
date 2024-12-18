import argparse
import asyncio
import random
import textwrap
import time

import gradio as gr
import xxhash
from dotenv import load_dotenv
from transformers import pipeline

import talk_arena.streaming_helpers as sh
from talk_arena.db_utils import TinyThreadSafeDB


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Talk Arena Demo")
    parser.add_argument("--free_only", action="store_true", help="Only use free models")
    return parser.parse_args()


args = parse_args()

if gr.NO_RELOAD:  # Prevents Re-init during hot reloading
    # Transcription Disabled for Public Interface
    # asr_pipe = pipeline(
    #    task="automatic-speech-recognition",
    #    model="openai/whisper-large-v3-turbo",
    #    chunk_length_s=30,
    #    device="cuda:1",
    # )

    anonymous = True

    # Generation Setup
    diva_audio, diva = sh.api_streaming("WillHeld/DiVA-llama-3-v0-8b")
    qwen2_audio, qwen2 = sh.api_streaming("Qwen/Qwen2-Audio-7B-Instruct")
    pipelined_system, pipeline_model = sh.api_streaming("pipeline/meta-llama/Meta-Llama-3-8B-Instruct")
    if not args.free_only:
        gemini_audio, gemini_model = sh.gemini_streaming("models/gemini-1.5-flash")
        gpt4o_audio, gpt4o_model = sh.gpt4o_streaming("models/gpt4o")
        geminip_audio, geminip_model = sh.gemini_streaming("models/gemini-1.5-pro")
        gemini2_audio, gemini2_model = sh.gemini_streaming("models/gemini-2.0-flash-exp")
    typhoon_audio, typhoon_model = sh.api_streaming("scb10x/llama-3-typhoon-audio-8b-2411")

    competitor_info = [
        (sh.gradio_gen_factory(diva_audio, "DiVA Llama 3 8B", anonymous), "diva_3_8b", "DiVA Llama 3 8B"),
        (sh.gradio_gen_factory(qwen2_audio, "Qwen 2", anonymous), "qwen2", "Qwen 2 Audio"),
        (
            sh.gradio_gen_factory(pipelined_system, "Pipelined Llama 3 8B", anonymous),
            "pipe_l3.0",
            "Pipelined Llama 3 8B",
        ),
        (sh.gradio_gen_factory(typhoon_audio, "Typhoon Audio", anonymous), "typhoon_audio", "Typhoon Audio"),
    ]
    # Add paid models if flag is not set
    if not args.free_only:
        competitor_info += [
            (sh.gradio_gen_factory(gemini_audio, "Gemini 1.5 Flash", anonymous), "gemini_1.5f", "Gemini 1.5 Flash"),
            (sh.gradio_gen_factory(gpt4o_audio, "GPT4o", anonymous), "gpt4o", "GPT-4o"),
            (sh.gradio_gen_factory(geminip_audio, "Gemini 1.5 Pro", anonymous), "gemini_1.5p", "Gemini 1.5 Pro"),
            (sh.gradio_gen_factory(geminip_audio, "Gemini 2 Flash", anonymous), "gemini_2f", "Gemini 2 Flash"),
        ]

    resp_generators = [generator for generator, _, _ in competitor_info]
    model_shorthand = [shorthand for _, shorthand, _ in competitor_info]
    model_name = [full_name for _, _, full_name in competitor_info]
    all_models = list(range(len(model_shorthand)))


async def pairwise_response_async(audio_input, state, model_order):
    if audio_input == None:
        raise StopAsyncIteration(
            "",
            "",
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            state,
            audio_input,
            None,
            None,
            None,
        )
    spinner_id = 0
    spinners = ["◐ ", "◓ ", "◑", "◒"]
    spinner = spinners[0]
    gen_pair = [resp_generators[model_order[0]], resp_generators[model_order[1]]]
    latencies = [{}, {}]  # Store timing info for each model
    resps = [gr.Textbox(value="", info="", visible=False), gr.Textbox(value="", info="", visible=False)]

    error_in_model = False
    for order, generator in enumerate(gen_pair):
        start_time = time.time()
        first_token = True
        total_length = 0
        try:
            async for local_resp in generator(audio_input, order):
                total_length += 1
                if first_token:
                    latencies[order]["time_to_first_token"] = time.time() - start_time
                    first_token = False
                resps[order] = local_resp
                spinner = spinners[spinner_id]
                spinner_id = (spinner_id + 1) % 4
                yield (
                    gr.Button(
                        value=spinner + " Generating Responses " + spinner,
                        interactive=False,
                        variant="primary",
                    ),
                    resps[0],
                    resps[1],
                    gr.Button(visible=False),
                    gr.Button(visible=False),
                    gr.Button(visible=False),
                    state,
                    audio_input,
                    None,
                    None,
                    latencies,
                )
            latencies[order]["total_time"] = time.time() - start_time
            latencies[order]["response_length"] = total_length
        except:
            error_in_model = True
            resps[order] = gr.Textbox(
                info=f"<strong>Error thrown by Model {order+1} API</strong>",
                value="" if first_token else resps[order]._constructor_args[0]["value"],
                visible=True,
                label=f"Model {order+1}",
            )
            yield (
                gr.Button(
                    value=spinner + " Generating Responses " + spinner,
                    interactive=False,
                    variant="primary",
                ),
                resps[0],
                resps[1],
                gr.Button(visible=False),
                gr.Button(visible=False),
                gr.Button(visible=False),
                state,
                audio_input,
                None,
                None,
                latencies,
            )
        latencies[order]["total_time"] = time.time() - start_time
        latencies[order]["response_length"] = total_length
    print(latencies)
    yield (
        gr.Button(value="Vote for which model is better!", interactive=False, variant="primary", visible=False),
        resps[0],
        resps[1],
        gr.Button(visible=not error_in_model),
        gr.Button(visible=not error_in_model),
        gr.Button(visible=not error_in_model),
        responses_complete(state),
        audio_input,
        gr.Textbox(visible=False),
        gr.Audio(visible=False),
        latencies,
    )


def on_page_load(state, model_order):
    if state == 0:
        # gr.Info(
        #    "Record something you'd say to an AI Assistant! Think about what you usually use Siri, Google Assistant,"
        #    " or ChatGPT for."
        # )
        state = 1
        model_order = random.sample(all_models, 2) if anonymous else model_order
    return state, model_order


def recording_complete(state):
    if state == 1:
        # gr.Info(
        #    "Once you submit your recording, you'll receive responses from different models. This might take a second."
        # )
        state = 2
    return (
        gr.Button(value="Starting Generation", interactive=False, variant="primary"),
        state,
    )


def responses_complete(state):
    if state == 2:
        gr.Info(
            "Give us your feedback! Mark which model gave you the best response so we can understand the quality of"
            " these different voice assistant models."
        )
        state = 3
    return state


def clear_factory(button_id):
    async def clear(audio_input, model_order, pref_counter, reasoning, latency):
        textbox1 = gr.Textbox(visible=False)
        textbox2 = gr.Textbox(visible=False)
        if button_id != None:
            sr, y = audio_input
            x = xxhash.xxh32(bytes(y)).hexdigest()
            await db.insert(
                {
                    "audio_hash": x,
                    "outcome": button_id,
                    "model_a": model_shorthand[model_order[0]],
                    "model_b": model_shorthand[model_order[1]],
                    "why": reasoning,
                    "model_a_latency": latency[0],
                    "model_b_latency": latency[1],
                }
            )
            pref_counter += 1
            model_a = model_name[model_order[0]]
            model_b = model_name[model_order[1]]
            textbox1 = gr.Textbox(
                visible=True,
                info=f"<strong style='color: #53565A'>Response from {model_a}</strong><p>Time-to-First-Character: {latency[0]['time_to_first_token']:.2f} ms, Time Per Character: {latency[0]['total_time']/latency[0]['response_length']:.2f} ms</p>",
            )
            textbox2 = gr.Textbox(
                visible=True,
                info=f"<strong style='color: #53565A'>Response from {model_b}</strong><p>Time-to-First-Character: {latency[1]['time_to_first_token']:.2f} ms, Time Per Character: {latency[1]['total_time']/latency[1]['response_length']:.2f} ms</p>",
            )

        try:
            sr, y = audio_input
            x = xxhash.xxh32(bytes(y)).hexdigest()
            os.remove(f"{x}.wav")
        except:
            # file already deleted, this is just a failsafe to assure data is cleared
            pass
        counter_text = f"# {pref_counter}/10 Preferences Submitted"
        if pref_counter >= 10 and False:  # Currently Disabled, Manages Prolific Completionx
            code = "PLACEHOLDER"
            counter_text = f"# Completed! Completion Code: {code}"
        counter_text = ""
        if anonymous:
            model_order = random.sample(all_models, 2)
        return (
            model_order,
            gr.Button(
                value="Record Audio to Submit Again!",
                interactive=False,
                visible=True,
            ),
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            None,
            textbox1,
            textbox2,
            pref_counter,
            counter_text,
            gr.Textbox(visible=False),
            gr.Audio(visible=False),
        )

    return clear


def transcribe(transc, voice_reason):
    if transc is None:
        transc = ""
    transc += " " + asr_pipe(voice_reason, generate_kwargs={"task": "transcribe"}, return_timestamps=False)["text"]
    return transc, gr.Audio(value=None)


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

with open("src/talk_arena/styles.css", "r") as css_file:
    custom_css = css_file.read()

db = TinyThreadSafeDB("live_votes.json")

with gr.Blocks(theme=theme, fill_height=True, css=custom_css) as demo:
    submitted_preferences = gr.State(0)
    state = gr.State(0)
    model_order = gr.State([])
    latency = gr.State([])
    with gr.Row():
        counter_text = gr.Markdown(
            ""
        )  # "# 0/10 Preferences Submitted.\n Follow the pop-up tips to submit your first preference.")
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], streaming=False, label="Audio Input")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            out1 = gr.Textbox(visible=False, lines=5, autoscroll=True)
        with gr.Column(scale=1):
            out2 = gr.Textbox(visible=False, lines=5, autoscroll=True)

    with gr.Row():
        btn = gr.Button(value="Record Audio to Submit!", interactive=False)

    with gr.Row(equal_height=True):
        reason = gr.Textbox(label="[Optional] Explain Your Preferences", visible=False, scale=4)
        reason_record = gr.Audio(
            sources=["microphone"],
            interactive=True,
            streaming=False,
            label="Speak to transcribe!",
            visible=False,
            type="filepath",
            # waveform_options={"show_recording_waveform": False},
            scale=1,
        )

    with gr.Row():
        best1 = gr.Button(value="Model 1 is better", visible=False)
        tie = gr.Button(value="Tie", visible=False)
        best2 = gr.Button(value="Model 2 is better", visible=False)

    with gr.Row():
        contact = gr.Markdown("")

    # reason_record.stop_recording(transcribe, inputs=[reason, reason_record], outputs=[reason, reason_record])
    audio_input.stop_recording(
        recording_complete,
        [state],
        [btn, state],
    ).then(
        fn=pairwise_response_async,
        inputs=[audio_input, state, model_order],
        outputs=[btn, out1, out2, best1, best2, tie, state, audio_input, reason, reason_record, latency],
    )
    audio_input.start_recording(
        lambda: gr.Button(value="Uploading Audio to Cloud", interactive=False, variant="primary"),
        None,
        btn,
    )
    best1.click(
        fn=clear_factory(0),
        inputs=[audio_input, model_order, submitted_preferences, reason, latency],
        outputs=[
            model_order,
            btn,
            best1,
            best2,
            tie,
            audio_input,
            out1,
            out2,
            submitted_preferences,
            counter_text,
            reason,
            reason_record,
        ],
    )
    tie.click(
        fn=clear_factory(0.5),
        inputs=[audio_input, model_order, submitted_preferences, reason, latency],
        outputs=[
            model_order,
            btn,
            best1,
            best2,
            tie,
            audio_input,
            out1,
            out2,
            submitted_preferences,
            counter_text,
            reason,
            reason_record,
        ],
    )
    best2.click(
        fn=clear_factory(1),
        inputs=[audio_input, model_order, submitted_preferences, reason, latency],
        outputs=[
            model_order,
            btn,
            best1,
            best2,
            tie,
            audio_input,
            out1,
            out2,
            submitted_preferences,
            counter_text,
            reason,
            reason_record,
        ],
    )
    audio_input.clear(
        clear_factory(None),
        [audio_input, model_order, submitted_preferences, reason, latency],
        [
            model_order,
            btn,
            best1,
            best2,
            tie,
            audio_input,
            out1,
            out2,
            submitted_preferences,
            counter_text,
            reason,
            reason_record,
        ],
    )
    demo.load(fn=on_page_load, inputs=[state, model_order], outputs=[state, model_order])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=40, api_open=False).launch(share=True, ssr_mode=False)
