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
