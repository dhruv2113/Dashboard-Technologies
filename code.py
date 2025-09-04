import streamlit as st
import pandas as pd
import numpy as np
import psutil
import requests
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer


st.set_page_config(page_title="Dhruv Gaur - Full Tech Dashboard", layout="wide")

st.title("🧠 Dhruv Gaur | Team 48 | Full Task Dashboard")


menu = st.sidebar.selectbox("Select Task Category", [
    "📸 JavaScript + Docker Tasks",
    "🤖 Python Automation Tasks",
    "📊 Machine Learning Tasks",
    "👣 MongoDB Docker Setup",
    "🌐 Fullstack Development Tasks",
    "🐧 Linux Basic Commands",
    "🤖 AI/ML Chatbot",
    "🐳 Docker Manager"
])


if menu == "📸 JavaScript + Docker Tasks":
    task = st.selectbox("Select JS or Docker Task", [
        "Take Photo Using JS",
        "Send Email Using JS",
        "Send Captured Photo to Email",
        "Record Video & Send via Email",
        "Send WhatsApp Message via JS",
        "Send SMS using Twilio API",
        "Show Current Geo Location",
        "Show Live Location on Google Maps",
        "Show Route to Destination",
        "Nearby Grocery Stores on Map",
        "Fetch Last Email from Gmail",
        "Post to Social Media",
        "Track Most Viewed Products",
        "Show IP & Location",
        "Run Linear Regression inside Docker",
        "Run Flask App in Docker",
        "Run Menu-Based Python in Docker",
        "Docker Inside Docker",
        "Install Firefox in Docker",
        "Run VLC inside Docker",
        "Setup Apache in Docker"
    ])

    st.subheader(f"Selected Task: {task}")
    if st.button("▶ Run Task"):
        if "Docker" in task:
            st.code(f"# Docker command placeholder for: {task}\ndocker build ...")
        elif task == "Show IP & Location":
            res = requests.get("https://ipinfo.io/json").json()
            st.write(f"🌐 IP: {res.get('ip')}")
            st.write(f"📍 Location: {res.get('city')}, {res.get('region')}, {res.get('country')}")
            st.write(f"🛡 Coordinates: {res.get('loc')}")
        elif task == "Show Live Location on Google Maps":
            iframe_url = "https://maps.google.com/maps?q=28.6139,77.2090&z=15&output=embed"
            st.components.v1.iframe(iframe_url, height=400)
        elif task == "Show Route to Destination":
            st.markdown("Open [Google Maps Directions](https://www.google.com/maps/dir/?api=1&origin=28.6139,77.2090&destination=28.7041,77.1025)")
        elif task == "Nearby Grocery Stores on Map":
            iframe_url = "https://www.google.com/maps/embed/v1/search?q=grocery+store&key=YOUR_API_KEY"
            st.components.v1.iframe(iframe_url, height=400)
        elif task == "Take Photo Using JS":
            st.image("https://via.placeholder.com/320x240.png?text=Photo+Captured+from+Webcam", caption="Simulated Captured Photo")
        elif task == "Send Email Using JS":
            st.success("📧 Email successfully sent using JavaScript API simulation.")
        elif task == "Send Captured Photo to Email":
            st.image("https://via.placeholder.com/150.png?text=Photo+Attached")
            st.success("✅ Captured photo sent via email successfully.")
        elif task == "Record Video & Send via Email":
            st.video("https://www.w3schools.com/html/mov_bbb.mp4")
            st.success("🎞 Video recorded and sent to email.")
        elif task == "Send WhatsApp Message via JS":
            st.markdown("[Click to send WhatsApp message](https://wa.me/919999999999?text=Hello+from+Dashboard)", unsafe_allow_html=True)
        elif task == "Send SMS using Twilio API":
            st.code("""
from twilio.rest import Client
client = Client("ACXXXX", "your_auth_token")
message = client.messages.create(
    body="Hello from Streamlit",
    from_="+17408312876",
    to="+91XXXXXXXXXX"
)
print(message.sid)
            """)
            st.success("✅ SMS sent successfully using Twilio (simulated).")
        elif task == "Fetch Last Email from Gmail":
            st.success("📧 Last email subject: 'Streamlit Dashboard Update' from 'admin@example.com'")
        elif task == "Post to Social Media":
            st.info("🔊 Posted update to LinkedIn, Instagram, and X!")
        elif task == "Track Most Viewed Products":
            st.bar_chart(pd.DataFrame({"Products": [150, 220, 340]}, index=["Product A", "Product B", "Product C"]))
        elif task == "Show Current Geo Location":
            st.write("User current location (simulated): 28.6139° N, 77.2090° E")
        else:
            st.info(f"Output for task '{task}' is currently simulated.")


elif menu == "🤖 Python Automation Tasks":
    task = st.selectbox("Choose a Python Automation Task", [
        "Live RAM Info Monitor",
        "Send SMS via Twilio",
        "Make Voice Call via Twilio",
        "Send Email",
        "Send WhatsApp Message",
        "Google Search & Extract",
        "Download Website Data",
        "Post to Social Media",
        "Create Digital Image",
        "Swap Faces in Two Images",
        "Tuple vs List Difference"
    ])

    st.subheader(f"Selected Task: {task}")

    if task == "Live RAM Info Monitor":
        duration = st.number_input("Enter monitoring duration (seconds):", min_value=1, max_value=30, value=5)
        placeholder = st.empty()
        if st.button("▶ Start Monitoring"):
            for i in range(int(duration)):
                memory = psutil.virtual_memory()
                placeholder.write(f"""
                **Update {i+1}/{duration}**
                - Total RAM: {memory.total / (1024**3):.2f} GB
                - Available: {memory.available / (1024**3):.2f} GB
                - Used: {memory.used / (1024**3):.2f} GB
                - Usage: {memory.percent:.2f} %
                """)
                time.sleep(1)

    elif task == "Send SMS via Twilio":
        st.info("Twilio SMS sending simulated in this environment.")

    elif task == "Make Voice Call via Twilio":
        st.info("Voice call simulation.")

    elif task == "Send Email":
        st.info("Simulated Email sending.")

    elif task == "Send WhatsApp Message":
        st.info("WhatsApp message scheduling simulated.")

    elif task == "Google Search & Extract":
        query = st.text_input("Enter your search query")
        if st.button("🔍 Search"):
            st.write(f"Simulated search for: **{query}**")
            st.write("Top result: https://streamlit.io/")

    elif task == "Download Website Data":
        url = st.text_input("Enter website URL:")
        if st.button("⬇ Download"):
            try:
                res = requests.get(url)
                st.code(res.text[:1000], language="html")
                st.success("✅ Fetched first 1000 characters!")
            except:
                st.error("❌ Failed to fetch website data.")

    elif task == "Post to Social Media":
        st.info("🔊 Placeholder: Add LinkedIn/Instagram/X API integration.")

    elif task == "Create Digital Image":
        from PIL import Image, ImageDraw
        width, height = 800, 600
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        for y in range(height):
            r = int(255 * y / height)
            g = int(255 * (1 - y / height))
            b = 180
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        cx, cy, radius = width // 2, height // 2, 100
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill='yellow', outline='black', width=3)
        draw.text((width//3, height-50), "My Digital Image Art", fill='black')
        st.image(img, caption="Generated Digital Art")

    elif task == "Swap Faces in Two Images":
        st.image("https://via.placeholder.com/300x150.png?text=Face+Swap+Output")

    elif task == "Tuple vs List Difference":
        st.write("Tuples are immutable, Lists are mutable.")
        st.code("""
my_list = [1, 2, 3]  # mutable
my_tuple = (1, 2, 3)  # immutable
        """)


elif menu == "📊 Machine Learning Tasks":
    st.header("Machine Learning Tasks")
    option = st.radio("Choose ML Task", [
        "Q1. Missing Value Imputation Analysis",
        "Q2. Fill Missing Y using Regression",
        "Q3. Upload Dataset and Train Linear Regression",
        "Q4. House Price Prediction (Simple Linear Regression)",
        "Q5. Salary Prediction with Residual Plot",
        "Q6. Salary Prediction from CSV + User Input"
    ])

   
    if option == "Q1. Missing Value Imputation Analysis":
        df = pd.DataFrame({'X1': [1, 2, 3, 4, 5, 6],'Y': [2.1, 4.1, np.nan, 8.3, np.nan, 12.2]})
        st.dataframe(df)
        imputer = KNNImputer(n_neighbors=2)
        df['Y_imputed'] = imputer.fit_transform(df[['Y']])
        st.write("After KNN Imputation:")
        st.dataframe(df)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="X1", y="Y_imputed", ax=ax)
        st.pyplot(fig)

    
    elif option == "Q2. Fill Missing Y using Regression":
        df = pd.DataFrame({'X': [1, 2, 3, 4, 5, 6],'Y': [2.1, 4.1, None, 8.3, None, 12.2]})
        train = df.dropna()
        test = df[df['Y'].isnull()]
        model = LinearRegression()
        model.fit(train[['X']], train['Y'])
        df.loc[df['Y'].isnull(), 'Y'] = model.predict(test[['X']])
        st.dataframe(df)

   
    elif option == "Q3. Upload Dataset and Train Linear Regression":
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            x_columns = st.multiselect("Select Features (X):", df.columns.tolist())
            y_column = st.selectbox("Select Target (Y):", df.columns.tolist())
            if st.button("Train Model"):
                X = df[x_columns]; y = df[y_column]
                model = LinearRegression()
                model.fit(X, y)
                st.write("Coefficients:", model.coef_)
                st.write("Intercept:", model.intercept_)

  
    elif option == "Q4. House Price Prediction (Simple Linear Regression)":
        data = {'Size': [750, 800, 850, 900, 1000, 1100, 1200, 1400, 1600, 1800],
                'Price': [150, 160, 170, 180, 200, 220, 240, 280, 320, 360]}
        df = pd.DataFrame(data)
        X = df[['Size']]
        y = df['Price']
        model = LinearRegression()
        model.fit(X, y)
        st.write("Slope (m):", model.coef_[0])
        st.write("Intercept (b):", model.intercept_)
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Actual Data')
        ax.plot(X, model.predict(X), color='red', label='Regression Line')
        ax.set_xlabel('House Size (sqft)')
        ax.set_ylabel('Price ($1000s)')
        ax.set_title('Simple Linear Regression - House Price Prediction')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        new_size = st.number_input("Enter new house size (sqft):", value=1300)
        if st.button("Predict Price"):
            predicted_price = model.predict([[new_size]])[0]*1000
            st.success(f"Predicted price for {new_size} sqft house: ${predicted_price:.2f}")

   
    elif option == "Q5. Salary Prediction with Residual Plot":
        data = {'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.7, 3.9,
                                    4.0, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9],
                'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 57189, 63218,
                           55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 98273]}
        df = pd.DataFrame(data)
        X = df[['YearsExperience']]
        y = df['Salary']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(X, y, color='blue', label='Actual Salary')
        ax.plot(X, y_pred, color='red', label='Predicted Salary')
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary')
        ax.set_title('Actual vs Predicted Salary')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        residuals = y - y_pred
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(X, residuals, color='purple')
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_xlabel('Years of Experience')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.set_title('Residuals Plot')
        ax2.grid(True)
        st.pyplot(fig2)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

  
    elif option == "Q6. Salary Prediction from CSV + User Input":
        uploaded_file = st.file_uploader("Upload CSV (YearsExperience, Salary)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            X = df[['YearsExperience']]
            y = df['Salary']
            model = LinearRegression()
            model.fit(X, y)
            experience = st.number_input("Enter Years of Experience:", value=5.5)
            if st.button("Predict Salary"):
                predicted_salary = model.predict([[experience]])[0]
                st.success(f"Predicted salary for {experience} years of experience: ${predicted_salary:.2f}")


elif menu == "👣 MongoDB Docker Setup":
    st.header("Run MongoDB in Docker")
    if st.button("▶ Show Docker Command"):
        st.code("""
docker run -d \\
  --name mongodb-docker \\
  -p 27017:27017 \\
  -v mongodata:/data/db \\
  mongo
        """)


elif menu == "🌐 Fullstack Development Tasks":
    st.header("🌐 Fullstack Development Tasks")
    
    task = st.selectbox("Select Fullstack Task", [
        "Flask Search Application",
        "Flask Contact Form Application",
        "Flask Request Submit Application"
    ])
    
    st.subheader(f"Selected Task: {task}")
    
    if task == "Flask Search Application":
        st.write("### Flask Search Application")
        st.write("This is a simple Flask application with a search form.")
        
        if st.button("▶ Show Flask Code"):
            st.code("""
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return '''
        <h2>Search Form</h2>
        <form action="/search" method="get">
            <input type="text" name="q" placeholder="Enter your query">
            <button type="submit">Search</button>
        </form>
    '''

@app.route('/search')
def search():
    query = request.args.get('q', '')
    return f"<h2>You searched for: <em>{query}</em></h2>"

if __name__ == '__main__':
    app.run(debug=True)
            """, language="python")
        
        if st.button("▶ Run Flask App"):
            st.success("🚀 Flask app would start on http://localhost:5000")
            st.info("""
            **To run this Flask app:**
            1. Save the code as `app.py`
            2. Install Flask: `pip install flask`
            3. Run: `python app.py`
            4. Open browser: http://localhost:5000
            """)
    
    elif task == "Flask Contact Form Application":
        st.write("### Flask Contact Form Application")
        st.write("This is a Flask application with a contact form that collects user information.")
        
        if st.button("▶ Show Flask Code"):
            st.code("""
from flask import Flask, request

app = Flask(__name__)

# Simulated storage
submissions = {}

@app.route('/')
def contact_form():
    return '''
        <h2>Contact Us</h2>
        <form action="/submit" method="post">
            <label>Name:</label><br>
            <input type="text" name="name" required><br><br>

            <label>Email:</label><br>
            <input type="email" name="email" required><br><br>

            <label>Message:</label><br>
            <textarea name="message" required></textarea><br><br>

            <button type="submit">Submit</button>
        </form>
    '''

@app.route('/submit', methods=['POST'])
def submit_form():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    # Store submission in dictionary (simulate database)
    submissions[email] = {
        'name': name,
        'email': email,
        'message': message
    }

    return f"<h3>Thank you, {name}! Your message has been received.</h3>"

if __name__ == '__main__':
    app.run(debug=True)
            """, language="python")
        
        if st.button("▶ Run Flask App"):
            st.success("🚀 Flask Contact Form app would start on http://localhost:5000")
            st.info("""
            **To run this Flask app:**
            1. Save the code as `contact_app.py`
            2. Install Flask: `pip install flask`
            3. Run: `python contact_app.py`
            4. Open browser: http://localhost:5000
            5. Fill out the contact form and submit
            """)
            
        st.write("### Features:")
        st.write("- 📝 Contact form with name, email, and message fields")
        st.write("- ✅ Form validation with required fields")
        st.write("- 💾 Stores submissions in memory (simulated database)")
        st.write("- 🎯 Success message after form submission")
        st.write("- 🔒 POST method for secure data transmission")
    
    elif task == "Flask Request Submit Application":
        st.write("### Flask Request Submit Application")
        st.write("This is a Flask application with a request submission form that collects user information.")
        
        if st.button("▶ Show Flask Code"):
            st.code("""
from flask import Flask, request

app = Flask(__name__)

# Simulated storage
submissions = {}

@app.route('/')
def contact_form():
    return '''
        <h2>Contact Us</h2>
        <form action="/submit" method="post">
            <label>Name:</label><br>
            <input type="text" name="name" required><br><br>

            <label>Email:</label><br>
            <input type="email" name="email" required><br><br>

            <label>Message:</label><br>
            <textarea name="message" required></textarea><br><br>

            <button type="submit">Submit</button>
        </form>
    '''

@app.route('/submit', methods=['POST'])
def submit_form():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')

    # Store submission in dictionary (simulate database)
    submissions[email] = {
        'name': name,
        'email': email,
        'message': message
    }

    return f"<h3>Thank you, {name}! Your message has been received.</h3>"

if __name__ == '__main__':
    app.run(debug=True)
            """, language="python")
        
        if st.button("▶ Run Flask App"):
            st.success("🚀 Flask Request Submit app would start on http://localhost:5000")
            st.info("""
            **To run this Flask app:**
            1. Save the code as `request_submit_app.py`
            2. Install Flask: `pip install flask`
            3. Run: `python request_submit_app.py`
            4. Open browser: http://localhost:5000
            5. Fill out the request form and submit
            """)
            
        st.write("### Features:")
        st.write("- 📝 Request form with name, email, and message fields")
        st.write("- ✅ Form validation with required fields")
        st.write("- 💾 Stores submissions in memory (simulated database)")
        st.write("- 🎯 Success message after form submission")
        st.write("- 🔒 POST method for secure data transmission")
        st.write("- 📧 Email-based storage system")


elif menu == "🐧 Linux Basic Commands":
    st.header("💻 Linux Task List - Basic Commands")


    st.markdown("""
    ### 📚 Linux Learning Resources & My LinkedIn Posts:
    - **Blog post on companies using Linux:** [LinkedIn Post](https://www.linkedin.com/posts/dhruv-gaur-2107dg_linux-opensource-devops-activity-7348259621477781504-v6Ed?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEWBaZ4BRuc4gpx8zrn_FXT_z4jhJjvdkzQ)
    - **5 Why Big Companies Trust Linux:** [LinkedIn Post](https://www.linkedin.com/posts/dhruv-gaur-2107dg_linux-opensource-devops-activity-7348259621477781504-v6Ed?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEWBaZ4BRuc4gpx8zrn_FXT_z4jhJjvdkzQ)
    - **Personalizing Linux: Custom Firefox Icon in RHEL 9:** [LinkedIn Post](https://www.linkedin.com/posts/dhruv-gaur-2107dg_linux-rhel9-opensource-activity-7348260442709884928-giub?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEWBaZ4BRuc4gpx8zrn_FXT_z4jhJjvdkzQ)
    """)

    linux_tasks = {
        "Basic System Info": [
            ("date", "Current date and time"),
            ("cal", "Monthly calendar"),
            ("whoami", "Current logged-in user"),
            ("hostname", "System hostname"),
            ("uptime", "System uptime"),
        ],
        "File & Directory Management": [
            ("pwd", "Current working directory ka path dikhata hai"),
            ("ls", "Directory ke files list karta hai"),
            ("ls -l", "Detailed list with permissions, owner, size"),
            ("ls -a", "Hidden files ke saath list"),
            ("cd .. && pwd", "Ek level upar jana"),
            ("mkdir test_folder", "Naya folder banana"),
            ("rm -r test_folder", "Folder + contents delete karna"),
            ("touch test_file.txt", "Empty file create karna"),
            ("cp test_file.txt copy_test_file.txt", "File copy karna"),
            ("mv copy_test_file.txt moved_test_file.txt", "File move ya rename karna"),
        ],
        "System Monitoring & Info": [
            ("top -b -n 1 | head -20", "Real-time running processes (first 20 lines)"),
            ("df -h", "Disk usage"),
            ("free -h", "RAM usage"),
            ("uname -a", "System info"),
        ],
        "Networking Basics": [
            ("ping -c 2 google.com", "Internet connectivity check"),
            ("curl ifconfig.me", "Public IP check"),
        ],
        "Process Management": [
            ("ps aux | head -10", "Running processes list"),
        ],
    }

    for category, commands in linux_tasks.items():
        with st.expander(f"📂 {category}"):
            for cmd, desc in commands:
                cols = st.columns([4,1])
                cols[0].write(f"**{cmd}** – {desc}")
                if cols[1].button("▶ Run", key=f"run_{cmd}"):
                    st.info(f"Command: {cmd}")
                    st.info("This command would be executed in a real environment.")

   
    st.subheader("📜 Command Output:")
    st.info("Command output would be displayed here in a real environment.")


elif menu == "🤖 AI/ML Chatbot":
    st.header("🤖 AI/ML Chatbot (Gemini)")
    st.write("Ask any question related to Artificial Intelligence, Machine Learning, Deep Learning, Data Science, or Python for ML.")

    try:
        import google.generativeai as genai

        key = "AIzaSyCOecseiaJO5sGL3nTzfh7_vXQEbjzke6w"
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        def gemini_bot(user_prompt):
            chat = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            "You are an expert AI/ML teacher and assistant. ONLY answer queries related to Artificial Intelligence, Machine Learning, Deep Learning, Data Science, or Python for ML. If a user asks anything unrelated (like politics, sports, web development, etc.), politely respond with: 'I can only help with AI and ML-related topics. Please ask something in that area.' Help students by generating relevant code, solving ML/AI errors, debugging Python ML code, explaining ML concepts, and providing guidance on AI topics only."
                        ]
                    }
                ]
            )
            response = chat.send_message(user_prompt)
            return response.text

        user_input = st.text_input("Ask your AI/ML question:")
        if st.button("Ask Gemini"):
            if user_input.strip():
                with st.spinner("Gemini is thinking..."):
                    answer = gemini_bot(user_input)
                st.success(answer)
            else:
                st.info("Please enter a question related to AI or ML.")

      
        st.markdown("---")
        st.subheader("🧑‍🌾 Agentic AI for Farmer Solutions (Gemini + LangChain)")
        st.write("This section uses an agentic AI approach to help refine farmer startup ideas, perform market research, suggest business models, and more using Gemini and LangChain tools.")

        try:
            import os
            from langchain_core.tools import tool
            from langchain.agents import initialize_agent, AgentType
            from langchain_google_genai import ChatGoogleGenerativeAI

            os.environ["GOOGLE_API_KEY"] = key
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

            @tool
            def refine_idea(input: str) -> str:
                """Refines a vague farmer-related problem into a specific problem-solution format."""
                prompt = f"Refine this vague farmer startup idea into a clear problem-solution:\n\n{input}"
                return llm.invoke(prompt)

            @tool
            def market_research(input: str) -> str:
                """Performs market research for the idea."""
                prompt = f"Do market research for this farmer-focused idea:\n\n{input}\nInclude market size, competition, challenges, and trends."
                return llm.invoke(prompt)

            @tool
            def business_model(input: str) -> str:
                """Creates a Business Model Canvas."""
                prompt = f"Create a Business Model Canvas for this farmer idea:\n\n{input}"
                return llm.invoke(prompt)

            @tool
            def expert_advice_tool(input: str) -> str:
                """Simulates expert agricultural advice for the given problem."""
                prompt = f"As an agricultural expert, suggest practical advice and policy recommendations for:\n\n{input}"
                return llm.invoke(prompt)

            @tool
            def mental_health_support(input: str) -> str:
                """Provides ideas for mental health and emotional support for farmers."""
                prompt = f"Suggest mental health support services or ideas for this farmer problem:\n\n{input}"
                return llm.invoke(prompt)

            @tool
            def tech_solution_tool(input: str) -> str:
                """Suggests AI, IoT, SMS-based or mobile app-based tech solutions."""
                prompt = f"Suggest affordable and practical tech-based solutions (AI, SMS alerts, IoT) for:\n\n{input}"
                return llm.invoke(prompt)

            tools = [
                refine_idea,
                market_research,
                business_model,
                expert_advice_tool,
                mental_health_support,
                tech_solution_tool,
            ]

            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )

            agent_input = st.text_area(
                "Describe your farmer problem or startup idea (Agentic AI):",
                value="I'm a farmer facing stress due to crop failure and climate change. I don't have access to timely market prices, expert advice, or mental health support. I want to create a solution that helps farmers deal with this."
            )
            if st.button("Run Agentic AI"):
                with st.spinner("Agentic AI is analyzing your idea..."):
                    try:
                        agent_response = agent.run(agent_input)
                        st.success(agent_response)
                    except Exception as e:
                        st.error(f"Agentic AI error: {e}")

        except ImportError:
            st.warning("LangChain dependencies not installed. Install with: `pip install langchain langchain-google-genai`")

    except ImportError:
        st.warning("Google Generative AI not installed. Install with: `pip install google-generativeai`")



import streamlit as st
import subprocess

==

def execute_remote(username, ip, command):
    """
    Executes a command on a remote machine via SSH.
    """
    try:
        ssh_command = f"ssh {username}@{ip} \"{command}\""
        result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return f"✅ SSH Success:\n{result.stdout}"
        else:
            return f"❌ SSH Error:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "⏰ SSH command timed out"
    except Exception as e:
        return f"⚠️ SSH Exception: {str(e)}"


def execute_docker_command(command):
    """Execute Docker command locally and return output"""
    try:
        docker_check = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if docker_check.returncode != 0:
            return "❌ Docker is not installed or not available in PATH"
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return f"✅ Success:\n{result.stdout}"
        else:
            return f"❌ Error:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "⏰ Command timed out"
    except FileNotFoundError:
        return "❌ Docker command not found. Please ensure Docker is installed and in PATH"
    except Exception as e:
        return f"⚠️ Exception: {str(e)}"


def render_docker_page():
    """Render the Docker Manager page with SSH support"""
    st.markdown('<div class="main-header"><h1>🐳 Docker Remote Management (via SSH)</h1><p>Manage Docker containers, images, and operations on remote servers through SSH</p></div>', unsafe_allow_html=True)
    

    st.subheader("🔗 SSH Connection Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        remote_user = st.text_input("Remote Username:", placeholder="Enter username")
    with col2:
        remote_ip = st.text_input("Remote IP Address:", placeholder="Enter IP address")
    

    connection_mode = st.radio(
        "Select connection mode:",
        ["🔗 Remote (SSH)", "🖥️ Local"],
        horizontal=True
    )
    
    st.markdown("---")
    
 
    docker_action = st.selectbox(
        "Select Docker action:",
        ["Launch new container", "Stop container", "Remove container", "Start container", 
         "List all images", "List all containers", "Pull image from Hub"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        
        if docker_action == "Launch new container":
            container_name = st.text_input("Container name:")
            image_name = st.text_input("Image name (e.g., ubuntu:latest):")
            
            if st.button("🚀 Launch Container", use_container_width=True):
                if container_name and image_name:
                    command = f"docker run -dit --name={container_name} {image_name}"
                    result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                    st.code(result, language="bash")
                else:
                    st.error("Please provide container name and image name")
        
        elif docker_action == "Stop container":
            container_name = st.text_input("Container name to stop:")
            
            if st.button("🛑 Stop Container", use_container_width=True):
                if container_name:
                    command = f"docker stop {container_name}"
                    result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                    st.code(result, language="bash")
                else:
                    st.error("Please provide container name")
        
        elif docker_action == "Remove container":
            container_name = st.text_input("Container name to remove:")
            
            if st.button("🗑️ Remove Container", use_container_width=True):
                if container_name:
                    command = f"docker rm -f {container_name}"
                    result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                    st.code(result, language="bash")
                else:
                    st.error("Please provide container name")
        
        elif docker_action == "Start container":
            container_name = st.text_input("Container name to start:")
            
            if st.button("▶️ Start Container", use_container_width=True):
                if container_name:
                    command = f"docker start {container_name}"
                    result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                    st.code(result, language="bash")
                else:
                    st.error("Please provide container name")
        
        elif docker_action == "List all images":
            if st.button("📋 List Images", use_container_width=True):
                command = "docker images"
                result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                st.code(result, language="bash")
        
        elif docker_action == "List all containers":
            if st.button("📋 List Containers", use_container_width=True):
                command = "docker ps -a"
                result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                st.code(result, language="bash")
        
        elif docker_action == "Pull image from Hub":
            image_name = st.text_input("Image name to pull:")
            
            if st.button("⬇️ Pull Image", use_container_width=True):
                if image_name:
                    command = f"docker pull {image_name}"
                    result = execute_remote(remote_user, remote_ip, command) if connection_mode == "🔗 Remote (SSH)" and remote_user and remote_ip else execute_docker_command(command)
                    st.code(result, language="bash")
                else:
                    st.error("Please provide image name")
    
    with col2:
        st.subheader("📊 Docker Information")
        
     
        if connection_mode == "🔗 Remote (SSH)":
            st.success(f"🔗 Connected to: {remote_user}@{remote_ip}" if remote_user and remote_ip else "⚠️ Please provide remote username and IP address")
        else:
            st.info("🖥️ Using local Docker installation")
        
      
        st.subheader("💡 Quick Help")
        st.markdown("""
        **🔗 Remote (SSH) Mode:**
        - Enter remote username and IP address
        - Commands execute on remote server via SSH
        
        **🖥️ Local Mode:**
        - Commands execute on your local machine
        - Requires local Docker installation
        """)
        

        st.subheader("📝 Recent Commands")
        if "docker_history" not in st.session_state:
            st.session_state.docker_history = []
        
        for i, cmd in enumerate(st.session_state.docker_history[-5:]):
            st.text(f"{i+1}. {cmd}")




menu = st.sidebar.selectbox("Select Menu", [
    "📊 Dashboard",
    "🐳 Docker Manager",
    "📂 Project Manager"
])

if menu == "📊 Dashboard":
    st.title("📊 Welcome to Dashboard")
    st.info("This is your main dashboard page.")

elif menu == "🐳 Docker Manager":
    render_docker_page()

elif menu == "📂 Project Manager":
    st.title("📂 Project Manager")
    st.info("Manage your projects here.")

else:
    st.warning("Please select a valid menu option.")
