import requests

# Replace text in json file below, and run python client.py to see prediction result and probability.
# jd = """In this role, you will have broad impact and exposure across Grammarly, working with team members from our Product, Research, Marketing, Engineering, and Finance teams.
#         Grammarly’s engineers and researchers have the freedom to innovate and uncover breakthroughs—and, in turn, influence our product roadmap. The complexity of our technical challenges is growing rapidly as we scale our interfaces, algorithms, and infrastructure. Read more about our stack or hear from our team on our technical blog .
#      """

jd = """The company is great, you will get a lot of money without working!!"""

response=requests.get("http://127.0.0.1:5000/predict", json={"text":jd})
print(response)
print(response.json())