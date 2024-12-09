# Running your Own Speech Arena

```sh
git clone git@github.com:SALT-NLP/talk-arena.git
cd talk-arena
pip install -e .
export GEMINI_API_KEY=$YOUR_KEY_HERE
export OPENAI_API_KEY=$YOUR_KEY_HERE
export API_MODEL_CONFIG=$JSON_STRINGFY_OF_API_CONFIG
python src/talk_arena/demo.py
```

Inference code at github.com/SALT-NLP/audiolm-inference-server


If you use this code for your own research, please cite us:
```bibtex
  @misc{talkarena2024,
      title={Talk Arena: Interactive Evaluation of Large Audio Models},
      author={Minzhi Li and Will Held and Michael Ryan and Kunat Pipatanakul and Potsawee Manakul and Hao Zhu and Diyi Yang},
      year={2024},
      url={talkarena.org}
    }
```

as well as the DiVA paper for which this code was originally developed:
```bibtex
@misc{DiVA,
      title={{D}istilling an {E}nd-to-{E}nd {V}oice {A}ssistant {W}ithout {I}nstruction {T}raining {D}ata}, 
      author={William Held and Minzhi Li and Michael Ryan and Weiyan Shi and Yanzhe Zhang and Diyi Yang},
      year={2024},
      eprint={2410.02678},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.02678}, 
}
```
