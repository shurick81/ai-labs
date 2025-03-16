# Trying LLM Google Gemini 1.5

1. Get a key from https://aistudio.google.com/app/apikey

2. Run a request:

```bash
GOOGLE_API_KEY=<set-key-value>;
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "What time is it now?"}]
    }]
   }'
```

# Trying LLM Google Gemini 2.0

1. Get a key from https://aistudio.google.com/app/apikey

2. Run a request:

```bash
GOOGLE_API_KEY=<set-key-value>;
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
  "contents": [{
    "parts":[{"text": "What time is it now in Stockholm?"}]
    }]
   }'
```

# Adding an image to the request

![Picture for analysis](image.jpg)

```bash
GOOGLE_API_KEY=<set-key-value>;
# Downloading a sample image
curl -o image.jpg "https://github.com/downloads/shurick81/ai-labs/lab-contents/002_first_machine_learning_experiments/image.jpg"

# Preparing the prompt
echo '{
  "contents":[
    {
      "parts":[
        {"text": "Describe the picture and list different object and activities that happen on the picture and also make predictions what we can expect will happen. Respond in the following format: \
        {description: description, objects: [object0, object1, object2, etc], activities: [activity0, activity1, activity2, etc], predictions: [prediction0, prediction1, prediction2, etc]}"},
        {
          "inline_data": {
            "mime_type":"image/jpeg",
            "data": "'$(base64 -i image.jpg)'"
          }
        }
      ]
    }
  ]
}' > request.json

# Requesting
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$GOOGLE_API_KEY" \
        -H 'Content-Type: application/json' \
        -d @request.json
```
