import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import googlemaps

API_KEY = "AIzaSyBhksGknvLc61HhxmoLTKw9DKxDE9Y__gE"
gmaps = googlemaps.Client(key=API_KEY)


def get_nearby_hospitals(lat, lng, radius=3000):
    try:
        places = gmaps.places_nearby(
            location=(lat, lng),
            radius=radius,
            type="hospital"
        )
        return places.get("results", [])
    except Exception as e:
        print(f"Error fetching hospitals: {e}")
        return []

 
def render():
    """Render the symptom-based disease predictor in a Streamlit tab."""
    # Load dataset
    data = pd.read_csv("./data/Symptom2Disease_with_care_AUTOFILLED.csv")
    data.columns = ["index", "disease", "symptoms", "care_instructions"]

    # Features and labels
    X = data["symptoms"]
    y = data["disease"]

    # Text vectorization
    vectorizer = CountVectorizer(stop_words="english")
    X_vectorized = vectorizer.fit_transform(X)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))

 
   

# Sidebar
    st.sidebar.image("./data/Logo.png")
    st.sidebar.write(
        "Aceso Care is an AI-powered health assistant that provides "
        "preliminary disease predictions based on user-described symptoms."
        "It does not replace a doctor.")

    st.sidebar.title("🏥 Find Nearby Hospitals")
        
    address = st.sidebar.text_input("📍 Enter your address (e.g., 'Bangalore, India')", 
                           placeholder="City, Country or full address",
                           key="address_input")
    lat = None
    lng = None
   
    if address:
        try:
            # Geocode the address
            geocode_result = gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                lat = location['lat']
                lng = location['lng']
                st.sidebar.success(f"📍 Location found: {lat:.6f}, {lng:.6f}")
            else:
                st.sidebar.warning("Could not find location for that address.")
        except Exception as e:
            st.sidebar.error(f"Error geocoding address: {e}")
    
    if st.sidebar.button("🔍 Search Hospitals"):
        if lat and lng:
            hospitals = get_nearby_hospitals(lat, lng)
            if hospitals:
                st.sidebar.write("**Nearby Hospitals:**")
                for h in hospitals:
                    name = h.get("name", "Unknown")
                    rating = h.get("rating", "N/A")
                    address = h.get("vicinity", "Address not available")
                    st.sidebar.write(f"🏥 **{name}** - Rating: {rating}")
                    st.sidebar.caption(f"📍 {address}")
                    st.sidebar.divider()
            else:
                st.sidebar.warning("No hospitals found nearby. Try expanding the search radius or check your location.")
        else:
            st.sidebar.error("Please provide a valid location (address or coordinates).")

  

    # Main input UI
    st.markdown(" Describe your symptoms")

    # Initialize user_input from session state if available (for speech recognition)
    

    # Text area for manual input or display recognized speech
    user_input = st.text_area(
        "Enter your symptoms",
        height=100
    )
    
    # Update session state with current input
    st.session_state.user_input = user_input
    single_symptom_rules = {
        "fever": ("Fever", "Take complete rest, drink fluids, and monitor temperature."),
        "headache": ("Headache", "Rest in a quiet room, hydrate well, reduce screen time."),
        "leg pain": ("Mucle Pain", "Take rest, avoid exertion, apply warm compress."),
        "cold": ("Common Cold", "Stay warm, drink warm fluids, take rest."),
        "cough": ("Cough", "Drink warm liquids, avoid cold food, rest your throat."),
        "vomiting": ("Vomiting", "Sip ORS slowly, avoid solid food temporarily."),
        "diarrhea": ("Diarrhea", "Drink ORS after every loose motion, rest."),
        "sore throat": ("Sore Throat", "Gargle with warm salt water, drink warm fluids, rest your voice."),
        "runny nose": ("Runny Nose", "Steam inhalation, stay warm, drink plenty of fluids."),
        "body pain": ("Body Ache", "Take adequate rest, apply warm compress, stay hydrated."),
        "fatigue": ("Fatigue", "Get enough sleep, eat balanced meals, avoid overexertion."),
        "dizziness": ("Dizziness", "Sit or lie down immediately, drink water, avoid sudden movements."),
        "stomach pain": ("Stomach Pain", "Eat light food, avoid spicy meals, rest."),
        "nausea": ("Nausea", "Take small sips of water or ginger tea, avoid oily food."),
        "constipation": ("Constipation", "Increase fiber intake, drink more water, stay active."),
        "acidity": ("Acidity", "Avoid spicy and fried food, eat small meals, do not lie down immediately after eating."),
        "shortness of breath": ("Breathlessness", "Sit upright, loosen tight clothing, avoid exertion."),
        "back pain": ("Back Pain", "Maintain proper posture, apply warm compress, avoid heavy lifting."),
        "joint pain": ("Joint Pain", "Rest the joint, apply warm compress, avoid strain."),
        "loss of appetite": ("Loss of Appetite", "Eat small frequent meals, choose light and nutritious food."),
        "dehydration": ("Dehydration", "Drink water or ORS frequently, avoid caffeine."),
        "chills": ("Chills", "Keep yourself warm, rest well, and drink warm fluids."),
        "sweating": ("Sweating", "Stay hydrated and wear loose, breathable clothing."),
        "blocked nose": ("Nasal Congestion", "Use steam inhalation and keep the head elevated."),
        "sneezing": ("Sneezing", "Avoid dust and allergens and keep surroundings clean."),
        "post nasal drip": ("Post Nasal Drip", "Drink warm fluids and use steam inhalation."),
        "watery eyes": ("Watery Eyes", "Rinse eyes gently with clean water and avoid rubbing."),
        "dry eyes": ("Dry Eyes", "Blink frequently and reduce screen exposure."),
        "ear blockage": ("Blocked Ear", "Avoid inserting objects and allow natural pressure equalization."),
        "neck pain": ("Neck Pain", "Maintain proper posture and apply warm compress."),
        "shoulder pain": ("Shoulder Pain", "Rest the shoulder and avoid heavy lifting."),
        "muscle stiffness": ("Muscle Stiffness", "Do gentle stretching and apply warm compress."),
        "muscle cramps": ("Muscle Cramps", "Gently stretch the muscle and stay hydrated."),
        "cold hands": ("Cold Hands", "Keep hands warm and improve circulation with movement."),
        "cold feet": ("Cold Feet", "Wear warm socks and keep feet dry."),
        "trembling": ("Trembling", "Sit calmly, relax, and take slow deep breaths."),
        "loss of appetite": ("Loss of Appetite", "Eat small, frequent meals and choose light foods."),
        "bitter taste": ("Bitter Taste in Mouth", "Rinse mouth regularly and stay hydrated."),
        "bad breath": ("Bad Breath", "Maintain oral hygiene and drink sufficient water."),
        "hiccups": ("Hiccups", "Sip cold water slowly and practice controlled breathing."),
        "burping": ("Excessive Burping", "Eat slowly and avoid fizzy drinks."),
        "abdominal bloating": ("Abdominal Bloating", "Avoid heavy meals and take short walks."),
        "general weakness": ("General Weakness", "Rest adequately and maintain balanced nutrition."),
        "difficulty concentrating": ("Difficulty Concentrating", "Take breaks, reduce stress, and ensure proper sleep.")
    }
    clean_input = user_input.lower().strip()

    if st.button("🔍 Analyze Symptoms"):
        if clean_input.strip() == "":
            st.warning("Please enter your symptoms.")
        elif clean_input in single_symptom_rules:
            prediction, care = single_symptom_rules[clean_input]
            st.success(f"🧾 Possible Condition: **{prediction}**")
            st.info("Basic Care Guidance: " + care)
        else:
            vector = vectorizer.transform([user_input])
            if vector.sum() == 0:
                st.warning("⚠️ Please explain your symptoms in more detail.")
            else:
                prediction = model.predict(vector)[0]
                # Use the correct column name defined above: "care_instructions"
                care = data[data["disease"] == prediction]["care_instructions"].iloc[0]
                st.success(f"🧾 Possible Condition: **{prediction}**")
                st.info("Basic Care Guidance: " + care)