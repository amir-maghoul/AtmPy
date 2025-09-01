from google import genai

client = genai.Client(api_key="AIzaSyDe3QRIu_Zy4I3olvxRicmTlJghDddmQhg")
myfile = client.files.upload(file=atmpy / "Cajun_instruments.jpg")

response = client.models.generate_content(
    model="gemini-2.5-pro-preview-05-06",
    contents="Tell me in few word what does my imex_time_integration.py file do",
)
print(response.text)
