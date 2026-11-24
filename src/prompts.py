PROMPTS = {
    "v1_basic": (
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "v2_grounded": (
        "Use only the context below to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer in one or two sentences."
    ),
    "v3_refuse": (
        "Use only the context below to answer the question. "
        "If the answer is not in the context, reply exactly: I don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
}