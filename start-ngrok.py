from pyngrok import ngrok
from app import app

if __name__ == "__main__":
    #Set your ngrok authtoken (get from https://dashboard.ngrok.com)
    ngrok.set_auth_token("33ggpDkzf0kcFAHHs5biPuZJ9vE_2fAaohx56bMsbiaMFZgW")

    #Start public tunnel for Flask's port (5000)
    public_url = ngrok.connect(5000)
    print(f" Public ngrok URL: {public_url}")

    #Save public URL to app config (so templates can use it)
    app.config["BASE_URL"] = public_url

    #Run your actual chatbot panel app
    app.run(port=5000, debug=False)
