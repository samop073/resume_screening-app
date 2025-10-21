"""
Resume Screening System - Streamlit Frontend
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from resume_screener import (
    process_multiple_resumes,
    rank_resumes,
    extract_skills
)

# Page config
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df_resumes' not in st.session_state:
    st.session_state.df_resumes = None
if 'ranked_results' not in st.session_state:
    st.session_state.ranked_results = None

# Header
st.markdown('<div class="main-header">📄 Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered resume matching with bias reduction</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    skill_weight = st.slider(
        "Skill Match Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="How much to weigh exact skill matches"
    )
    
    semantic_weight = 1.0 - skill_weight
    st.info(f"Semantic Match Weight: {semantic_weight:.1f}")
    
    top_k = st.number_input(
        "Number of Top Candidates",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system uses:
    - 🤖 NLP for text analysis
    - 🔒 Anonymization to reduce bias
    - 🎯 Semantic + skill matching
    - 📊 Embedding-based ranking
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "🏆 Results", "📊 Analytics"])

with tab1:
    st.header("Upload Resumes and Job Description")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload resume files (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="Upload one or more resume files"
        )
        
        if uploaded_resumes:
            st.success(f"✅ {len(uploaded_resumes)} resume(s) uploaded")
    
    with col2:
        st.subheader("📋 Job Description")
        job_desc = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Enter the full job description including required skills, experience, and qualifications..."
        )
        
        if job_desc:
            jd_skills = extract_skills(job_desc)
            st.info(f"**Required Skills Detected:** {', '.join(jd_skills) if jd_skills else 'None'}")
    
    st.markdown("---")
    
    # Process button
    if st.button("🚀 Process Resumes", type="primary", use_container_width=True):
        if not uploaded_resumes:
            st.error("⚠️ Please upload at least one resume")
        elif not job_desc:
            st.error("⚠️ Please enter a job description")
        else:
            with st.spinner("Processing resumes... This may take a minute."):
                try:
                    # Process resumes
                    df_resumes = process_multiple_resumes(uploaded_resumes)
                    
                    if len(df_resumes) == 0:
                        st.error("❌ No valid resumes could be processed. Check file formats.")
                    else:
                        st.session_state.df_resumes = df_resumes
                        
                        # Rank resumes
                        ranked = rank_resumes(
                            df_resumes,
                            job_desc,
                            top_k=top_k,
                            skill_weight=skill_weight,
                            semantic_weight=semantic_weight
                        )
                        st.session_state.ranked_results = ranked
                        
                        st.success(f"✅ Successfully processed {len(df_resumes)} resumes!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Error processing resumes: {str(e)}")

with tab2:
    st.header("Top Candidates")
    
    if st.session_state.ranked_results is not None and len(st.session_state.ranked_results) > 0:
        results = st.session_state.ranked_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Resumes", len(st.session_state.df_resumes))
        with col2:
            st.metric("Top Candidates", len(results))
        with col3:
            avg_score = results['final_score'].mean()
            st.metric("Avg Match Score", f"{avg_score:.2f}")
        with col4:
            top_score = results['final_score'].iloc[0]
            st.metric("Best Match", f"{top_score:.2f}")
        
        st.markdown("---")
        
        # Display ranked candidates
        for idx, row in results.iterrows():
            with st.expander(f"#{idx+1} - {row['filename']} (Score: {row['final_score']:.3f})", expanded=(idx<3)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Final Score", f"{row['final_score']:.3f}")
                with col2:
                    st.metric("Semantic Match", f"{row['semantic_score']:.3f}")
                with col3:
                    st.metric("Skill Match", f"{row['skill_score']:.3f}")
                
                st.markdown("**Skills Found:**")
                if row['skills']:
                    skills_display = ", ".join(row['skills'])
                    st.markdown(f"_{skills_display}_")
                else:
                    st.markdown("_No skills detected_")
                
                with st.expander("View Resume Text (Anonymized)"):
                    st.text(row['anonymized_text'][:2000] + "..." if len(row['anonymized_text']) > 2000 else row['anonymized_text'])
        
        # Download results
        st.markdown("---")
        csv = results[['filename', 'final_score', 'skill_score', 'semantic_score', 'skills']].to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👆 Upload resumes and process them in the 'Upload & Process' tab first")

with tab3:
    st.header("Analytics Dashboard")
    
    if st.session_state.ranked_results is not None and len(st.session_state.ranked_results) > 0:
        results = st.session_state.ranked_results
        
        # Score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution")
            score_data = pd.DataFrame({
                'Candidate': [f"#{i+1}" for i in range(len(results))],
                'Final Score': results['final_score'],
                'Semantic Score': results['semantic_score'],
                'Skill Score': results['skill_score']
            })
            st.bar_chart(score_data.set_index('Candidate'))
        
        with col2:
            st.subheader("Top Skills Across Candidates")
            all_skills = []
            for skills in results['skills']:
                all_skills.extend(skills)
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(10)
                st.bar_chart(skill_counts)
            else:
                st.info("No skills detected in top candidates")
        
        # Detailed stats
        st.markdown("---")
        st.subheader("Statistical Summary")
        
        stats_df = results[['final_score', 'semantic_score', 'skill_score']].describe()
        st.dataframe(stats_df, use_container_width=True)
        
    else:
        st.info("👆 Process resumes first to see analytics")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Built with Streamlit | NLP-powered Resume Screening System
</div>
""", unsafe_allow_html=True)
