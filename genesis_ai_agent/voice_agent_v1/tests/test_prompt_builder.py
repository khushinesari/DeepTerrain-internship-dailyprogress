from modules.prompt_builder import (
    build_prompt,
    save_prompt
)

transcript = "Increase mission priority to HIGH."

prompt = build_prompt(transcript)

save_prompt(prompt)

print(prompt)