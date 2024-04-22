import streamlit as st
import serpapi  
import openai
import time
import csv
from io import StringIO
import threading
from queue import Queue
import base64

# Initialize the OpenAI client with your API key
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]  )


system_message = "You answer a user's question, given some text as context to help answer the question. The user request will be followed by the context. The context given is from the user's Google search results, it is current and up to date. Do not contradict the contents of the given text in your answer."

# Function to fetch SERPAPI results
def fetch_serpapi_results(query, serpapi_key):
    params = {
        "engine": "google",
        "q": query,
        "api_key": serpapi_key  
    }
    search = serpapi.search(params)
    results = search.get_dict()
    
    # Extracting the organic results
    organic_results = results.get("organic_results", [])
    organic_text = ""
    for result in organic_results:
        title = result['title']
        snippet = result.get('snippet', '')
        
        # Extract rating and price information from rich_snippet
        rich_snippet = result.get('rich_snippet', {})
        top_info = rich_snippet.get('top', {})
        detected_extensions = top_info.get('detected_extensions', {})
        rating = detected_extensions.get('rating')
        price_range_from = detected_extensions.get('price_range_from')
        
        # Add rating and price information to the organic text if available
        if rating:
            organic_text += f"Rating: {rating}\n"
        if price_range_from:
            organic_text += f"Price range: from ${price_range_from}\n"
            
        organic_text += f"{title}: {snippet}\n\n"
        
    # Extracting the answer box or the desired snippet from organic results
    answer_text = ""
    answer_box = results.get("answer_box", {})
    if answer_box:
        answer_text = answer_box.get("answer", "")
    if not answer_text:
        # If the answer box doesn't have an "answer" key, 
        # look for the desired snippet in organic results.
        for result in organic_results:
            if 'organic_result' in result.get('type', ''):
                answer_text = result.get('snippet', '')
                break
            
    # If still no answer text, extract the snippet from the answer_box
    if not answer_text and answer_box:
        answer_text = answer_box.get('snippet', '')
        
    # Combining the organic text and answer box text
    text = f"{answer_text}\n{organic_text}"
    return text

def query_agent_stream(prompt, delay_time=0.01, speech=False):
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    )
    reply_content = ''
    # Assuming the response has a method or property to access the content directly
    try:
        # If the completion object has a method or property to get the content
        message_content = completion.choices[0].message.content  # Adjusted based on expected object structure
        print(message_content, end='', flush=True)
        reply_content += message_content
    except AttributeError:
        # Handle the case where the expected attributes are not found
        print("Error accessing message content from the completion response.")
        
    if reply_content and reply_content[-1] in {'.', '!', '?'}:
        time.sleep(delay_time)
    return reply_content

def ask_question(query, custom_prompt, serpapi_key):
    if query:
        # Fetch Google search results
        search_results = fetch_serpapi_results(query, serpapi_key)
        
        # Construct a new prompt with the search results appended
        query_with_context = custom_prompt + '\n' + search_results
        
        # Get the response from GPT-4
        response = query_agent_stream(query_with_context)
        
        return search_results, response  # Return both search_results and response
    return None, "Please enter a question."  # Return None for search_results if no query is entered

def generate_response(i, messages, custom_prompt, serpapi_key):
    # Extract the question from the messages
    question = messages[1]['content']
    # Fetch the search results and get the response using the ask_question function
    search_results, response = ask_question(question, custom_prompt, serpapi_key)
    # For this example, we'll return the index, response, and search results.
    # You can modify this to return other data if needed.
    return i, response, search_results

class WorkerThread(threading.Thread):
    def __init__(self, jobs, results, custom_prompt, serpapi_key):
        super().__init__()
        self.jobs = jobs
        self.results = results
        self.custom_prompt = custom_prompt
        self.serpapi_key = serpapi_key
        
    def run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                break
            i, messages = job
            result = generate_response(i, messages, self.custom_prompt, self.serpapi_key)
            self.results.put(result)
            
def create_download_link(data, filename):
    csv_data = StringIO()
    writer = csv.writer(csv_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Answers"])  # Add header row
    for row in data:
        writer.writerow([row])  # This is already correct, as it writes the entire string as a single field
    b64 = base64.b64encode(csv_data.getvalue().encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{filename}</a>'

def main():

    st.title("GPT Search Queries")
    openai_key = st.secrets["OPENAI_API_KEY"]
    serpapi_key = st.secrets["SERP_API_KEY"]
    st.sidebar.header("Queries")
    data_section = st.sidebar.text_area("CSV or Text Data:")
    paste_data = st.sidebar.button("Paste Data")
    custom_prompt = st.sidebar.text_area("Custom Prompt:", value="QUESTION? Provide me with a one or two-word answer. Be as succinct as possible.")
    num_concurrent_calls = st.sidebar.number_input("Concurrent Calls:", min_value=1, max_value=2000, value=10, step=1)
    generate_all = st.sidebar.button("Generate All")

    if openai_key:
        openai.api_key = openai_key 

    add_row = st.sidebar.button("Add row")
    reset = st.sidebar.button("Reset")

    row_count = st.session_state.get("row_count", 1)

    if paste_data:
        data = StringIO(data_section.strip())
        reader = csv.reader(data, delimiter='\n', quotechar='"')
        questions = [row[0] for row in reader]
        row_count = len(questions)
        for i, question in enumerate(questions):
            st.session_state[f"query_{i}"] = question
        st.session_state["row_count"] = row_count

    if add_row:
        row_count += 1
        st.session_state["row_count"] = row_count

    if reset:
        row_count = 1
        st.session_state.clear()
        st.session_state["row_count"] = row_count

    if generate_all:
        messages = [st.session_state.get(f"query_{i}", "") for i in range(row_count)]

        jobs = Queue()
        results = Queue()

        workers = [WorkerThread(jobs, results, custom_prompt, serpapi_key) for _ in range(num_concurrent_calls)]

        for worker in workers:
            worker.start()

        for i, message in enumerate(messages):
            jobs.put((i, [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ]))

        for _ in range(num_concurrent_calls):
            jobs.put(None)  # Signal the end of the job queue

        for worker in workers:
            worker.join()  # Wait for all worker threads to complete

        while not results.empty():
            i, response, search_results = results.get()
            st.session_state[f"response_{i}"] = response
            st.session_state[f"search_results_{i}"] = search_results

    for i in range(st.session_state.get("row_count", 1)):
        col1, col2, col3 = st.columns(3)

        with col1:
            query_key = f"query_{i}"
            query = st.text_area(f"Enter question {i + 1} here:", key=query_key, height=50, value=st.session_state.get(query_key, ""))

        with col2:
            st.text_area(f"Search Results {i + 1}", key=f"search_results_{i}", value=st.session_state.get(f"search_results_{i}"), height=50)

        with col3:
            st.text_area(f"Answer {i + 1}", key=f"response_{i}", value=st.session_state.get(f"response_{i}"), height=50)

        if st.button(f"Ask {i + 1}"): 
            if openai_key and serpapi_key:
                with st.spinner(f'Fetching the response for question {i + 1}...'):
                    search_results, response = ask_question(st.session_state.get(query_key, ""), custom_prompt, serpapi_key)
                    st.session_state[f"search_results_{i}"] = search_results
                    st.session_state[f"response_{i}"] = response
    
    responses_data = [st.session_state.get(f"response_{i}", "") for i in range(st.session_state.get("row_count", 1))]

    # Create a download link for a CSV file containing the answers
    download_responses_link_csv = create_download_link(responses_data, "Download Responses.csv")

    # Display the download link
    st.markdown(download_responses_link_csv, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    