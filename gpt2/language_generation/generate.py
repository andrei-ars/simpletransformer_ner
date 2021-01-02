import logging

from simpletransformers.language_generation import LanguageGenerationModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = LanguageGenerationModel("gpt2", "outputs/fine-tuned/", args={"max_length": 200}, use_cuda=False)
# model = LanguageGenerationModel("gpt2", "outputs/fine-tuned", args={"max_length": 200})
# model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 200})

prompts = [
    "Click on \"Mrs.\"",
    "Click on New Lead",
    "Click on secondary checkbox.",
]

for prompt in prompts:
    # Generate text using the model. Verbose set to False to prevent logging generated sequences.
    generated = model.generate(prompt, verbose=False)

    generated = ".".join(generated[0].split(".")[:-1]) + "."
    print("=============================================================================")
    print(generated)
    print("=============================================================================")
