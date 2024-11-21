from together import Together
import config

client = Together(api_key=config.TOGETHER_API_KEY)

response = client.images.generate(
    prompt="space robots", model="black-forest-labs/FLUX.1-schnell", steps=4
)

print(response.data[0].url)

