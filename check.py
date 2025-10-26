import google.generativeai as genai
genai.configure(api_key="AIzaSyC4zoBNvjVQXXfwA9t-gMlV0t4mLfJ64Lo")
for m in genai.list_models():
    print(m.name)