from langchain.llms import OpenAI

llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)

print(llm("Tell me a joke"))

llm_result = llm.generate(["Tell me a joke", "Tell me a poem"])

print(llm_result)
