import os
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts.chat import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time
import hashlib
from datetime import datetime, timedelta

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Enhanced Caching with TTL and Query Hashing
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def cache_query_response(query_hash, response_data):
    """Cache query responses with TTL"""
    return response_data

def get_query_hash(query, memory_context=""):
    """Generate hash for query + context for intelligent caching"""
    combined = f"{query.lower().strip()}{memory_context}"
    return hashlib.md5(combined.encode()).hexdigest()

@st.cache_resource(show_spinner=False, ttl=7200)  # Cache for 2 hours
def load_assets():
    """Load and cache retriever + embeddings with enhanced error handling"""
    from data_loader import load_documents
    from text_processing import create_embedding_model, create_vector_store

    DATA_PATH = "vit_dataset"
    try:
        with st.spinner("🧠 Loading knowledge base..."):
            documents = load_documents(DATA_PATH)
            embedding_model = create_embedding_model()
            retriever = create_vector_store(
                documents,
                embedding_model,
                "university",
                "./vectordb"
            )
            # Enhanced retrieval parameters for better performance
            retriever.search_kwargs = {
                "k": 6,  # Reduced from 8 for faster retrieval
            }
            return retriever
    except Exception as e:
        st.error(f"❌ Failed to load documents: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=False, ttl=7200)
def get_llm():
    """Load and cache the LLM with connection pooling"""
    from llm_model import initialize_llm
    with st.spinner("⚡ Initializing AI model..."):
        # Add connection pooling if using API-based LLM
        llm = initialize_llm()
        # Configure for faster responses
        if hasattr(llm, 'temperature'):
            llm.temperature = 0.3  # Lower temperature for faster, more consistent responses
        return llm

# -----------------------------
# Enhanced UI Components
# -----------------------------
def inject_advanced_css():
    """Inject advanced CSS with animations and modern design"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary: #4f46e5;
            --primary-light: #6366f1;
            --primary-dark: #3730a3;
            --secondary: #06b6d4;
            --accent: #f59e0b;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --user-bg: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            --assistant-bg: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            --sidebar: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            --glass: rgba(255, 255, 255, 0.25);
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.1), 0 4px 6px rgba(0,0,0,0.05);
            --shadow-xl: 0 20px 25px rgba(0,0,0,0.15), 0 10px 10px rgba(0,0,0,0.04);
        }
        
        * { font-family: 'Inter', sans-serif; }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            min-height: 100vh;
            padding-bottom: 120px;
        }
        
        /* Glassmorphism header */
        .header {
            text-align: center;
            padding: 2rem 1rem;
            margin-bottom: 2rem;
            position: sticky;
            top: 0;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            z-index: 1000;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: var(--shadow-lg);
            animation: slideDown 0.6s ease-out;
        }
        
        @keyframes slideDown {
            from { transform: translateY(-100%); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .header-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4f46e5, #06b6d4, #10b981);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: gradientShift 4s ease infinite;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .header-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.2rem;
            margin: 0.8rem 0 0;
            font-weight: 400;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Enhanced floating chat input */
        .stChatFloatingInputContainer {
            position: fixed !important;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: min(800px, 90vw);
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: var(--shadow-xl);
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            animation: fadeInUp 0.5s ease-out;
        }
        
        @keyframes fadeInUp {
            from { transform: translateX(-50%) translateY(100%); opacity: 0; }
            to { transform: translateX(-50%) translateY(0); opacity: 1; }
        }
        
        /* Enhanced chat messages */
        .stChatMessage {
            max-width: 80%;
            padding: 20px 24px;
            border-radius: 20px;
            margin-bottom: 24px;
            line-height: 1.7;
            font-size: 0.95rem;
            position: relative;
            animation: messageSlide 0.4s ease-out;
            word-wrap: break-word;
        }
        
        @keyframes messageSlide {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .stChatMessage.user {
            background: var(--user-bg);
            margin-left: auto;
            margin-right: 2%;
            border-bottom-right-radius: 6px;
            box-shadow: var(--shadow-md);
            border: 1px solid rgba(79, 70, 229, 0.2);
        }
        
        .stChatMessage.assistant {
            background: var(--assistant-bg);
            margin-left: 2%;
            margin-right: auto;
            border-bottom-left-radius: 6px;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(6, 182, 212, 0.1);
            position: relative;
        }
        
        .stChatMessage.assistant::before {
            content: '';
            position: absolute;
            left: -1px;
            top: 0;
            height: 100%;
            width: 4px;
            background: linear-gradient(180deg, var(--primary), var(--secondary));
            border-radius: 20px 0 0 20px;
        }
        
        /* Enhanced sidebar */
        .stSidebar > div {
            background: var(--sidebar) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(10px);
        }
        
        .sidebar-content {
            padding: 1.5rem 1rem;
        }
        
        .settings-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
            font-size: 1.4rem;
            font-weight: 600;
            color: var(--primary-dark);
        }
        
        /* Enhanced buttons */
        .stButton button {
            background: linear-gradient(135deg, var(--primary), var(--primary-light)) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        .stButton button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-lg) !important;
            background: linear-gradient(135deg, var(--primary-light), var(--secondary)) !important;
        }
        
        /* Enhanced typing indicator */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 16px 24px;
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(6, 182, 212, 0.1));
            border-radius: 20px;
            border-bottom-left-radius: 6px;
            margin-left: 2%;
            margin-right: auto;
            max-width: 80%;
            box-shadow: var(--shadow-md);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        .typing-text {
            color: var(--primary);
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            animation: typingBounce 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typingBounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }
        
        /* Enhanced metrics and stats */
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.8);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Enhanced expandable sections */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(5px) !important;
        }
        
        .streamlit-expanderContent {
            background: rgba(255, 255, 255, 0.02) !important;
            border-radius: 0 0 12px 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .status-online {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .status-processing {
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .header-title { font-size: 2.2rem; }
            .header-subtitle { font-size: 1rem; }
            .stChatMessage { max-width: 95%; }
            .stChatFloatingInputContainer { width: 95vw; }
        }
        
        /* Accessibility improvements */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --assistant-bg: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                --sidebar: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_enhanced_header():
    """Render enhanced header with real-time status"""
    current_time = datetime.now().strftime("%H:%M")
    st.markdown(f"""
    <div class="header">
        <h1 class="header-title">VIT AI Assistant</h1>
        <p class="header-subtitle">Your intelligent university guide powered by advanced AI</p>
        <div class="status-indicator status-online">
            <span>●</span> Online • {current_time}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_sidebar():
    """Render enhanced sidebar with advanced options"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("""
        <div class="settings-header">
            ⚙️ Advanced Settings
        </div>
        """, unsafe_allow_html=True)
        
        # Performance settings
        st.subheader("🚀 Performance")
        memory_window = st.slider(
            "Conversation Memory",
            min_value=2, max_value=24, value=8,
            help="Number of previous messages to remember"
        )
        
        use_cache = st.toggle(
            "Smart Caching",
            value=True,
            help="Cache responses for faster replies to similar questions"
        )
        
        # Display settings
        st.subheader("🎨 Display")
        show_context = st.toggle(
            "Show Source Documents",
            value=False,
            help="Display retrieved documents used for answers"
        )
        
        show_metrics = st.toggle(
            "Show Performance Metrics",
            value=False,
            help="Display response time and caching statistics"
        )
        
        # Response settings
        st.subheader("💬 Response Style")
        response_style = st.selectbox(
            "Response Detail Level",
            ["Concise", "Balanced", "Detailed"],
            index=1,
            help="Choose how detailed you want the responses"
        )
        
        # Actions
        st.subheader("🔧 Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        
        with col2:
            clear_cache = st.button("🧹 Clear Cache", use_container_width=True)
        
        if clear_cache:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
            time.sleep(1)
            st.rerun()
        
        # Statistics
        if show_metrics:
            st.subheader("📊 Session Stats")
            chat_count = len(st.session_state.get('chat_history', []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{chat_count//2}</div>
                <div class="metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.8rem; line-height: 1.6;">
            <p><strong>VIT AI Assistant</strong></p>
            <p>Powered by LangChain & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'memory_window': memory_window,
            'show_context': show_context,
            'show_metrics': show_metrics,
            'response_style': response_style,
            'use_cache': use_cache,
            'clear_chat': clear_chat
        }

def show_enhanced_typing_indicator():
    """Show enhanced typing indicator"""
    return st.markdown("""
    <div class="typing-indicator">
        <span class="typing-text">VIT Assistant is thinking</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Enhanced Processing Functions
# -----------------------------
def get_optimized_prompt(response_style):
    """Get optimized prompt based on response style"""
    
    style_configs = {
        "Concise": {
            "instruction": "Provide brief, direct answers. Use bullet points when listing items.",
            "max_length": "Keep responses under 150 words."
        },
        "Balanced": {
            "instruction": "Provide comprehensive yet concise responses. Balance detail with readability.",
            "max_length": "Aim for 150-300 words."
        },
        "Detailed": {
            "instruction": "Provide thorough, detailed explanations with examples and context.",
            "max_length": "Use 300-500 words when necessary."
        }
    }
    
    config = style_configs[response_style]
    
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are VIT's official AI assistant with enhanced capabilities. 

        <context>
        {{context}}
        </context>
        
        **Response Guidelines:**
        - {config['instruction']}
        - {config['max_length']}
        - Use clear headings and structure when appropriate
        - Include relevant emojis sparingly for better readability
        - For unknown information: "I don't have that specific information. Please contact VIT administration at ap.admissions@vitap.ac.in"
        - Prioritize accuracy over completeness
        
        **Current conversation context:**
        {{chat_history}}"""),
        ("human", "{input}")
    ])

async def process_query_with_caching(query, retrieval_chain, settings):
    """Process query with intelligent caching and performance optimization"""
    start_time = time.time()
    
    # Generate query hash for caching
    memory_context = str(st.session_state.chat_history[-settings['memory_window']:]) if settings['memory_window'] > 0 else ""
    query_hash = get_query_hash(query, memory_context)
    
    # Check cache if enabled
    if settings['use_cache']:
        try:
            # Try to get cached response
            cached_response = st.session_state.get(f"cache_{query_hash}")
            if cached_response and (time.time() - cached_response['timestamp']) < 1800:  # 30 min cache
                if settings['show_metrics']:
                    st.success(f"✨ Retrieved from cache in {(time.time() - start_time)*1000:.0f}ms")
                return cached_response['data']
        except:
            pass
    
    # Process query
    try:
        response = retrieval_chain.invoke({
            "input": query,
            "chat_history": st.session_state.chat_history[-settings['memory_window']:]
        })
        
        # Cache the response if enabled
        if settings['use_cache']:
            st.session_state[f"cache_{query_hash}"] = {
                'data': response,
                'timestamp': time.time()
            }
        
        # Show metrics if enabled
        if settings['show_metrics']:
            processing_time = (time.time() - start_time) * 1000
            st.info(f"⚡ Processed in {processing_time:.0f}ms • Sources: {len(response.get('context', []))}")
        
        return response
        
    except Exception as e:
        st.error(f"⚠️ Processing error: {str(e)}")
        return {"answer": "I apologize, but I encountered an error processing your request. Please try again."}

def display_enhanced_context(docs, show_context):
    """Display context documents with enhanced UI"""
    if show_context and docs:
        with st.expander("📚 **Source Documents**", expanded=False):
            st.markdown("**Documents used for this response:**")
            
            for i, doc in enumerate(docs[:3]):
                relevance_score = doc.metadata.get('score', 0)
                source = doc.metadata.get('source', 'Unknown')
                
                # Color code by relevance
                color = "#10b981" if relevance_score > 0.8 else "#f59e0b" if relevance_score > 0.6 else "#ef4444"
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; margin-bottom: 1rem; border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong>📄 Document {i+1}</strong>
                        <span style="background: {color}; color: white; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.7rem;">
                            {relevance_score:.2f}
                        </span>
                    </div>
                    <p style="font-size: 0.8rem; color: rgba(255,255,255,0.7); margin-bottom: 0.8rem;">📂 {source}</p>
                    <p style="font-size: 0.9rem; line-height: 1.5;">{doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}</p>
                </div>
                """, unsafe_allow_html=True)
    elif show_context:
        st.info("💡 No specific source documents were used for this general response.")

# -----------------------------
# Main Enhanced Application
# -----------------------------
def main():
    # Enhanced page configuration
    st.set_page_config(
        page_title="VIT AI Assistant - Enhanced",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'mailto:ap.admissions@vitap.ac.in',
            'Report a bug': 'mailto:support@vitap.ac.in',
            'About': "VIT AI Assistant"
        }
    )
    
    # Inject advanced CSS
    inject_advanced_css()
    
    # Render enhanced header
    render_enhanced_header()
    
    # Initialize enhanced session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.session_state.cache_hits = 0
    
    # Enhanced sidebar
    settings = render_enhanced_sidebar()
    
    # Clear chat if requested
    if settings['clear_chat']:
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.session_state.cache_hits = 0
        st.rerun()
    
    
    # Load resources with enhanced error handling
    try:
        with st.spinner("🚀 Initializing your enhanced AI assistant..."):
            retriever = load_assets()
            llm = get_llm()
    except Exception as e:
        st.error(f"⚠️ Critical initialization error: {str(e)}")
        st.info("Please check your configuration and try refreshing the page.")
        st.stop()
    
    if retriever is None or llm is None:
        st.error("⚠️ Failed to initialize AI components. Please contact support.")
        st.stop()
    
    # Create optimized chains
    qa_prompt = get_optimized_prompt(settings['response_style'])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Display enhanced chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(f"**You**  \n{msg.content}")
        else:
            with st.chat_message("assistant", avatar="🏫"):
                st.markdown(f"**VIT Assistant**  \n{msg.content}")
    
    # Enhanced chat input with suggestions
    placeholder_text = "Ask about VIT courses, admissions, campus life, or anything else..."
    
    if prompt := st.chat_input(placeholder_text):
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.total_queries += 1
        
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"**You**  \n{prompt}")
        
        # Process with enhanced typing indicator
        with st.chat_message("assistant", avatar="🏫"):
            with st.empty():
                show_enhanced_typing_indicator()
                
                # Process query with caching and optimization
                try:
                    response = retrieval_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history[-settings['memory_window']:]
                    })
                    
                    answer = response.get("answer", "I couldn't generate a response. Please try again.")
                    
                    # Display response
                    st.markdown(f"**VIT Assistant**  \n{answer}")
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    # Display enhanced context
                    display_enhanced_context(response.get("context", []), settings['show_context'])
                    
                except Exception as e:
                    error_msg = f"⚠️ **Processing Error**  \nI encountered an issue: {str(e)}\n\nPlease try rephrasing your question or contact support."
                    st.markdown(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))

if __name__ == "__main__":
    main()