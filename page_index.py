from pageindex import PageIndexClient
 
pi_client = PageIndexClient(api_key="b8a8e69469824245b8077aa72f992c0c")

result = pi_client.submit_document("./data/course_catalog.pdf")
doc_id = result["doc_id"]

response = pi_client.chat_completions(
    messages=[{"role": "user", "content": "give me information about the Computer Architecture course?"}],
    doc_id=doc_id
)
 
print(response["choices"][0]["message"]["content"])