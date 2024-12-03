import asyncio
import random
import textwrap
import time

import gradio as gr
import xxhash
from transformers import pipeline

import talk_arena.streaming_helpers as sh
from talk_arena.db_utils import TinyThreadSafeDB


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
    gemini_audio, gemini_model = sh.gemini_streaming("models/gemini-1.5-flash")
    gpt4o_audio, gpt4o_model = sh.gpt4o_streaming("models/gpt4o")
    geminip_audio, geminip_model = sh.gemini_streaming("models/gemini-1.5-pro")
    typhoon_audio, typhoon_model = sh.api_streaming("scb10x/llama-3-typhoon-audio-8b-2411")

    competitor_info = [
        (sh.gradio_gen_factory(diva_audio, "DiVA Llama 3 8B", anonymous), "diva_3_8b"),
        (
            sh.gradio_gen_factory(qwen2_audio, "Qwen 2", anonymous),
            "qwen2",
        ),
        (
            sh.gradio_gen_factory(pipelined_system, "Pipelined Llama 3 8B", anonymous),
            "pipe_l3.0",
        ),
        (
            sh.gradio_gen_factory(gemini_audio, "Gemini 1.5 Flash", anonymous),
            "gemini_1.5f",
        ),
        (
            sh.gradio_gen_factory(gpt4o_audio, "GPT4o", anonymous),
            "gpt4o",
        ),
        (
            sh.gradio_gen_factory(geminip_audio, "Gemini 1.5 Pro", anonymous),
            "gemini_1.5p",
        ),
        (
            sh.gradio_gen_factory(typhoon_audio, "Typhoon Audio", anonymous),
            "typhoon_audio",
        ),
    ]
    resp_generators = [generator for generator, _ in competitor_info]
    model_shorthand = [shorthand for _, shorthand in competitor_info]
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
    gen_pair = [resp_generators[model_order[0]], resp_generators[model_order[1]]]
    latencies = [{}, {}]  # Store timing info for each model
    resps = ["", ""]
    for order, generator in enumerate(gen_pair):
        start_time = time.time()
        first_token = True
        total_length = 0
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
    print(latencies)
    yield (
        gr.Button(value="Click to compare models!", interactive=True, variant="primary"),
        resps[0],
        resps[1],
        gr.Button(visible=True),
        gr.Button(visible=True),
        gr.Button(visible=True),
        responses_complete(state),
        audio_input,
        gr.Textbox(visible=False),
        gr.Audio(visible=False),
        latencies,
    )


def on_page_load(state, model_order):
    if state == 0:
        gr.Info(
            "Record something you'd say to an AI Assistant! Think about what you usually use Siri, Google Assistant,"
            " or ChatGPT for."
        )
        state = 1
        model_order = random.sample(all_models, 2) if anonymous else model_order
    return state, model_order


def recording_complete(state):
    if state == 1:
        gr.Info(
            "Once you submit your recording, you'll receive responses from different models. This might take a second."
        )
        state = 2
    return (
        gr.Button(value="Click to compare models!", interactive=True, variant="primary"),
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
    def clear(audio_input, model_order, pref_counter, reasoning, latency):
        if button_id != None:
            sr, y = audio_input
            x = xxhash.xxh32(bytes(y)).hexdigest()
            db.insert(
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

        try:
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
                value="Record Audio to Submit!",
                interactive=False,
            ),
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            None,
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
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

db = TinyThreadSafeDB("user_study.json")
with gr.Blocks(theme=theme, fill_height=True) as demo:
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

    with gr.Row():
        btn = gr.Button(value="Record Audio to Submit!", interactive=False)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            out1 = gr.Textbox(visible=False, max_lines=5, lines=5)
        with gr.Column(scale=1):
            out2 = gr.Textbox(visible=False, max_lines=5, lines=5)

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
        contact = gr.Markdown(
            textwrap.dedent(
                """
        <details>
        <summary>Rubric To Judge Model Outputs</summary>
        Please select the response which is better among the two candidates by considering the following dimensions:
        
        - Naturalness: Evaluates how closely the response resembles human language, focusing on fluency, grammar, and appropriate tone to ensure it sounds natural.
        
        - Coherence: Assesses the logical flow and clarity of ideas, ensuring that each part of the response contributes meaningfully and is internally consistent.
        
        - Groundedness: Ensure the response is related to the input and is based on accurate, reliable information, avoiding unsupported or speculative statements to enhance credibility.
        
        - Harmlessness: Choose the response that sounds most similar to what a peaceful, ethical, and respectful person would say.
        
        - Helpfulness: Choose the response that has the most actionable information relevant to the inputs provided or which most completely follows the instructions given.
        </details>
        <details>
        <summary>Contact Information and IRB Info</summary>
        
        ## What We Store
        
        This platform does not store any information about you or your queries for further research or release. All recordings are erased after you submit your votes and vote data is stored only in aggregate to understand model performance.

        This platform sends data to third-party APIs from OpenAI and Google. In both cases, we utilize paid APIs for which the terms of service dictate that your data will not be used for training or stored for more than 30 days.

        Based on the lack of identifiable information and the focus on model rankings, the Stanford IRB has determined that this public platform is not human subects research as defined in 45 CFR 46.102(e).

        ## CONTACT INFORMATION:

        *Questions:* If you have any questions, concerns or complaints about this research, its procedures, risks and benefits, contact the Diyi Yang, diyiy@cs.stanford.edu. 

        *Independent Contact:* If you are not satisfied with how this study is being conducted, or if you have any concerns, complaints, or general questions about the research or your rights as a participant, please contact the Stanford Institutional Review Board (IRB) to speak to someone independent of the research team at 650-723-2480 or toll free at 1-866-680-2906, or email at irbnonmed@stanford.edu. You can also write to the Stanford IRB, Stanford University, 1705 El Camino Real, Palo Alto, CA 94306.
         </details>
            """
            )
        )

    # reason_record.stop_recording(transcribe, inputs=[reason, reason_record], outputs=[reason, reason_record])
    audio_input.stop_recording(
        recording_complete,
        [state],
        [btn, state],
    )
    audio_input.start_recording(
        lambda: gr.Button(value="Uploading Audio to Cloud", interactive=False, variant="primary"),
        None,
        btn,
    )
    btn.click(
        fn=pairwise_response_async,
        inputs=[audio_input, state, model_order],
        outputs=[btn, out1, out2, best1, best2, tie, state, audio_input, reason, reason_record, latency],
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
    demo.queue(default_concurrency_limit=40, api_open=False).launch(share=True)
