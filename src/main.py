from retrieval import load_index, prompt_llm, retrieve_chunks


def main():
    question = input("Enter a question: ")
    print("Answer: ", pipeline(question))


def pipeline(question):
    index, id_map = load_index()
    chunks = retrieve_chunks(question, index, id_map)
    return prompt_llm(chunks, question)


if __name__ == "__main__":
    main()
