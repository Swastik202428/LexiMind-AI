"""Composition and presentation for the LexiMind AI Streamlit pages."""

from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from modules import (
    chatbot,
    classification,
    embeddings,
    keywords,
    language,
    ner,
    pos_tagging,
    preprocessing,
    question_answering,
    sentiment,
    statistics as text_statistics,
    summarization,
    tokenization,
    topic_modeling,
)
from ui.components import (
    render_downloads,
    render_module_heading,
    render_text_workspace,
    save_analysis,
    set_source_text,
    show_saved_result,
)
from utils.helpers import record_analysis

PAGES = [
    ("🏠  Dashboard", "Dashboard", "Your NLP workspace at a glance."),
    ("📝  Text Preprocessing", "Text Preprocessing", "Normalize text with configurable cleaning steps."),
    ("🔤  Tokenization & N-Grams", "Tokenization & N-Grams", "Explore sentences, tokens, and phrase patterns."),
    ("📊  Text Statistics", "Text Statistics", "Measure vocabulary, frequency, and text shape."),
    ("💜  Sentiment Analysis", "Sentiment Analysis", "Estimate opinion polarity and subjectivity."),
    ("🏷️  Named Entity Recognition", "Named Entity Recognition", "Find people, organizations, places, dates, and money."),
    ("📚  Text Classification", "Text Classification", "Predict spam status or a news category."),
    ("🔎  Keyword Extraction", "Keyword Extraction", "Rank useful terms using TF-IDF and frequency."),
    ("📖  Text Summarization", "Text Summarization", "Create a compact, extractive summary."),
    ("🌍  Language Detection", "Language Detection", "Identify the likely language of a text."),
    ("🧩  POS Tagging", "POS Tagging", "Inspect grammatical roles token by token."),
    ("🧱  Lexical Vectors", "Lexical Vectors", "Explore count, TF-IDF, and local context vectors."),
    ("📌  Topic Modeling", "Topic Modeling", "Discover themes across a set of documents."),
    ("💬  Question Answering", "Question Answering", "Retrieve the most relevant sentence from context."),
    ("🤖  NLP Chatbot", "NLP Chatbot", "Ask a small, transparent NLP-concepts bot."),
]
PAGE_BY_LABEL = {label: title for label, title, _ in PAGES}
SAMPLE_TEXT = (
    "Natural language processing helps computers work with human language. "
    "Researchers use tokenization, sentiment analysis, and machine learning to discover "
    "useful patterns in text. LexiMind AI brings these practical NLP techniques together "
    "in one accessible workspace. The platform can analyze documents, identify key terms, "
    "and summarize important ideas."
)
SAMPLE_DOCUMENTS = (
    "Researchers build language tools with tokenization, text classification, and machine learning. "
    "NLP software can identify entities and estimate sentiment in customer feedback.\n"
    "A company reported strong quarterly revenue after launching a new product. Investors watched "
    "market growth, sales, and business strategy.\n"
    "The football team won the final match after a season of training. Coaches praised the players "
    "and fans celebrated the championship."
)


def render_dashboard(on_navigate) -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Natural Language Processing Workspace</div>'
        '<h1 class="main-title">Understand language.<br>Unlock intelligence.</h1>'
        '<div class="subtitle">A practical, privacy-friendly toolkit for turning text into insight.</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    with st.container(key="dashboard_metrics"):
        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("NLP modules", len(PAGES) - 1)
        metric_cols[1].metric("Words processed", f"{st.session_state['total_words_processed']:,}")
        metric_cols[2].metric("Analyses run", f"{st.session_state['analysis_count']:,}")
        metric_cols[3].metric("Current session", "Local & privacy-friendly")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Your NLP pipeline")
        stages = ["Input", "Clean", "Features", "Model", "Insights"]
        positions = list(range(len(stages)))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=[0] * len(stages),
                mode="lines",
                line={"color": "rgba(155, 123, 255, .58)", "width": 3},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[(index + index + 1) / 2 for index in range(len(stages) - 1)],
                y=[0] * (len(stages) - 1),
                mode="text",
                text=["›"] * (len(stages) - 1),
                textfont={"color": "#6dcdf0", "size": 22},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=[0] * len(stages),
                mode="markers+text",
                text=stages,
                textposition="bottom center",
                textfont={"color": "#e8eaf5", "size": 11},
                marker={
                    "size": 18,
                    "color": ["#4bc7ec", "#72b7f2", "#9b7bff", "#9b7bff", "#70d7cf"],
                    "line": {"color": "#151827", "width": 2},
                },
                customdata=[
                    "Submit source text",
                    "Normalize and tokenize",
                    "Build lexical features",
                    "Run an NLP method",
                    "Review the analysis result",
                ],
                hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            height=185,
            showlegend=False,
            xaxis={"visible": False, "range": [-0.35, len(stages) - 0.65], "fixedrange": True},
            yaxis={"visible": False, "range": [-0.55, 0.35], "fixedrange": True},
            margin={"l": 8, "r": 8, "t": 12, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel={"bgcolor": "#191c30", "font": {"color": "#e8eaf5"}},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        st.markdown("### Recent activity")
        history = st.session_state["analysis_history"]
        if history:
            for entry in history[:5]:
                st.markdown(
                    f"**{escape(entry['module'])}** · {entry['time']}  \n"
                    f"<span style='color:#9da3bd'>{entry['words']} words · {escape(entry['preview'])}</span>",
                    unsafe_allow_html=True,
                )
                st.divider()
        else:
            st.info("Your recent analyses will appear here. Choose a module below to get started.")
    st.markdown("### Explore modules")
    for start in range(0, len(PAGES) - 1, 3):
        cards = st.columns(3)
        for column, (label, title, description) in zip(cards, PAGES[start + 1 : start + 4]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### {label.split('  ')[0]} {title}")
                    st.caption(description)
                    st.button(
                        f"Open {title}",
                        key=f"open_{title}",
                        on_click=on_navigate,
                        args=(label,),
                        width="stretch",
                    )
    st.markdown("### Session report")
    render_downloads()


def _render_preprocessing(text: str) -> None:
    st.markdown("Configure your normalization pipeline, then compare the original and cleaned text.")
    options = st.columns(4)
    lowercase = options[0].checkbox("Lowercase", value=True)
    punctuation = options[1].checkbox("Remove punctuation", value=True)
    special = options[2].checkbox("Remove special characters", value=False)
    numbers = options[3].checkbox("Remove numbers", value=False)
    options2 = st.columns(3)
    stopwords = options2[0].checkbox("Remove stopwords", value=True)
    stemming = options2[1].checkbox("Apply stemming", value=False)
    lemmatization = options2[2].checkbox("Apply lemmatization", value=False)
    if st.button("Run preprocessing", type="primary", key="run_preprocessing"):
        try:
            result = preprocessing.process_text(
                text,
                lowercase=lowercase,
                remove_punctuation=punctuation,
                remove_special_characters=special,
                remove_numbers=numbers,
                remove_stopwords=stopwords,
                stemming=stemming,
                lemmatization=lemmatization,
            )
            save_analysis("Text Preprocessing", text, result, result["cleaned"])
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Text Preprocessing")
    if result:
        if result.get("resource_warning"):
            st.warning(result["resource_warning"])
        before, after = st.columns(2)
        with before:
            st.markdown("**Before**")
            st.text_area("Original text", result["original"], height=150, disabled=True, key="prep_before")
            st.metric("Words before", result["word_count_before"])
        with after:
            st.markdown("**After**")
            st.text_area("Cleaned text", result["cleaned"], height=150, disabled=True, key="prep_after")
            st.metric("Words after", result["word_count_after"])
        st.caption("Applied: " + (", ".join(result["operations"]) if result["operations"] else "no transformations"))
        st.download_button("Download cleaned text", result["cleaned"], "cleaned-text.txt", "text/plain")


def _render_tokenization(text: str) -> None:
    if st.button("Tokenize text", type="primary", key="run_tokenization"):
        try:
            result = tokenization.tokenize(text)
            result["ngrams"] = tokenization.ngram_counts(text)
            save_analysis("Tokenization & N-Grams", text, result, f"{result['word_count']} word tokens")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Tokenization & N-Grams")
    if result:
        count_col, sentence_col = st.columns(2)
        count_col.metric("Word tokens", result["word_count"])
        sentence_col.metric("Sentences", len(result["sentences"]))
        st.markdown("**Sentences**")
        st.dataframe(pd.DataFrame({"Sentence": result["sentences"]}), width="stretch", hide_index=True)
        st.markdown("**Word tokens**")
        st.write(" · ".join(result["words"]))
        tabs = st.tabs(["Unigrams", "Bigrams", "Trigrams"])
        for tab, (label, counts) in zip(tabs, result["ngrams"].items()):
            with tab:
                rows = [{"N-gram": item, "Count": count} for item, count in counts.most_common(30)]
                if rows:
                    frame = pd.DataFrame(rows)
                    st.plotly_chart(
                        px.bar(frame.head(10), x="Count", y="N-gram", orientation="h", color="Count", color_continuous_scale=["#4bc7ec", "#9b7bff"]),
                        width="stretch",
                    )
                    st.dataframe(frame, width="stretch", hide_index=True)
                else:
                    st.info(f"Not enough tokens to form {label.lower()}.")


def _render_statistics(text: str) -> None:
    if st.button("Analyze statistics", type="primary", key="run_statistics"):
        try:
            result = text_statistics.analyze_text(text)
            result["source_text"] = text
            save_analysis("Text Statistics", text, result, f"{result['word_count']} words, {result['vocabulary_size']} unique")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Text Statistics")
    if result:
        columns = st.columns(5)
        columns[0].metric("Words", result["word_count"])
        columns[1].metric("Characters", result["character_count"])
        columns[2].metric("Sentences", result["sentence_count"])
        columns[3].metric("Vocabulary", result["vocabulary_size"])
        columns[4].metric("Avg. word length", f"{result['average_word_length']:.1f}")
        chart_col, cloud_col = st.columns(2)
        if result["frequencies"]:
            frame = pd.DataFrame(result["frequencies"], columns=["Word", "Frequency"])
            with chart_col:
                st.markdown("#### Most frequent words")
                chart = px.bar(frame, x="Frequency", y="Word", orientation="h", color="Frequency", color_continuous_scale=["#4bc7ec", "#9b7bff"])
                chart.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(chart, width="stretch")
            with cloud_col:
                st.markdown("#### Word cloud")
                try:
                    import matplotlib.pyplot as plt
                    from wordcloud import WordCloud

                    cloud = WordCloud(width=800, height=440, background_color="#111322", colormap="cool", max_words=100).generate(result["source_text"])
                    fig, axis = plt.subplots(figsize=(8, 4.4), facecolor="#111322")
                    axis.imshow(cloud, interpolation="bilinear")
                    axis.axis("off")
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                except ValueError:
                    st.info("Not enough distinct words to generate a word cloud.")


def _render_sentiment(text: str) -> None:
    if st.button("Analyze sentiment", type="primary", key="run_sentiment"):
        try:
            result = sentiment.analyze_sentiment(text)
            save_analysis("Sentiment Analysis", text, result, f"{result['label']} · polarity {result['polarity']:.2f}")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Sentiment Analysis")
    if result:
        left, right = st.columns([1, 1.4])
        with left:
            st.metric("Overall sentiment", result["label"])
            st.metric("Polarity", f"{result['polarity']:+.3f}", help="−1 is negative, 0 is neutral, +1 is positive.")
            st.metric("Subjectivity", f"{result['subjectivity']:.3f}", help="0 is more objective; 1 is more subjective.")
            st.caption(result["method"])
        with right:
            fig = go.Figure(go.Bar(
                x=["Polarity", "Subjectivity"],
                y=[result["polarity"], result["subjectivity"]],
                marker_color=["#4bc7ec" if result["polarity"] >= 0 else "#ef7195", "#9b7bff"],
            ))
            fig.update_layout(yaxis={"range": [-1, 1], "zeroline": True}, height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        st.caption("Lexicon-based estimate: sarcasm, context, and domain-specific meanings may not be captured.")


def _render_ner(text: str) -> None:
    if st.button("Find named entities", type="primary", key="run_ner"):
        try:
            result = ner.extract_entities(text)
            result["source_text"] = text
            save_analysis("Named Entity Recognition", text, result, f"{len(result['entities'])} entities · {result['method']}")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Named Entity Recognition")
    if result:
        source_text = result["source_text"]
        st.caption(result["method"])
        if result["fallback"]:
            st.info("spaCy's English model is not installed. These pattern-based results are approximate; install `en_core_web_sm` for model-based NER.")
        entities = result["entities"]
        if entities:
            entity_frame = pd.DataFrame([{"Entity": item["text"], "Type": item["label"]} for item in entities])
            table_col, chart_col = st.columns([1.4, 1])
            with table_col:
                st.dataframe(entity_frame, width="stretch", hide_index=True)
            with chart_col:
                counts = entity_frame["Type"].value_counts().rename_axis("Type").reset_index(name="Count")
                st.plotly_chart(px.bar(counts, x="Type", y="Count", color="Type"), width="stretch")
            chunks, cursor = [], 0
            for item in entities:
                if item["start"] < cursor:
                    continue
                chunks.append(escape(source_text[cursor : item["start"]]))
                chunks.append(
                    f"<mark style='background:rgba(155,123,255,.28);color:#e8eaf5;padding:2px 5px;border-radius:4px'>"
                    f"{escape(source_text[item['start']:item['end']])}<sup style='color:#66d8f1'> {escape(item['label'])}</sup></mark>"
                )
                cursor = item["end"]
            chunks.append(escape(source_text[cursor:]))
            st.markdown(
                "<div style='line-height:2.1;background:rgba(25,28,48,.82);padding:16px;border-radius:12px'>"
                + "".join(chunks) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No named entities were identified.")


def _render_classification(text: str) -> None:
    task = st.selectbox("Classification task", ["Spam vs. non-spam", "News category"], key="classification_task")
    task_key = "spam" if task.startswith("Spam") else "news"
    if st.button("Classify text", type="primary", key="run_classification"):
        try:
            result = classification.classify_text(text, task_key)
            result["task"] = task
            save_analysis("Text Classification", text, result, result["label"])
        except ValueError as error:
            st.warning(str(error))
        except RuntimeError as error:
            st.error(str(error))
    result = show_saved_result("Text Classification")
    if result:
        st.caption(f"Task: {result['task']}")
        st.metric("Predicted class", result["label"])
        st.progress(result["confidence"], text=f"Uncalibrated model score: {result['confidence']:.1%}")
        st.caption(result["confidence_note"])
        st.caption(
            f"{result['model']} · trained from {result['training_examples']} included examples. "
            "Held-out test metrics are shown below; this remains a demonstration baseline."
        )
        frame = pd.DataFrame([{"Class": label, "Probability": score} for label, score in result["scores"].items()])
        chart = px.bar(frame, x="Probability", y="Class", orientation="h", color="Probability", color_continuous_scale=["#4bc7ec", "#9b7bff"])
        chart.update_layout(xaxis={"range": [0, 1]}, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(chart, width="stretch")

        evaluation = result["evaluation"]
        test_metrics = evaluation["test"]
        st.markdown("#### Held-out evaluation")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{test_metrics['accuracy']:.1%}")
        metric_cols[1].metric("Precision", f"{test_metrics['precision_macro']:.1%}")
        metric_cols[2].metric("Recall", f"{test_metrics['recall_macro']:.1%}")
        metric_cols[3].metric("Macro-F1", f"{test_metrics['macro_f1']:.1%}")
        st.caption(
            f"{evaluation['dataset_source']} · "
            f"{evaluation['split']['strategy']} · random state {evaluation['random_state']}."
        )

        per_class_rows = [
            {
                "Class": label,
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["macro_f1"],
                "Support": values["support"],
            }
            for label, values in test_metrics["per_class"].items()
        ]
        st.dataframe(pd.DataFrame(per_class_rows), width="stretch", hide_index=True)

        confusion = pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=test_metrics["class_labels"],
            columns=test_metrics["class_labels"],
        )
        st.markdown("**Confusion matrix**")
        st.dataframe(confusion, width="stretch")

        comparison_rows = [
            {
                "Model": model_name,
                "Validation accuracy": values["validation"]["accuracy"],
                "Validation precision": values["validation"]["precision_macro"],
                "Validation recall": values["validation"]["recall_macro"],
                "Validation macro-F1": values["validation"]["macro_f1"],
            }
            for model_name, values in evaluation["model_comparison"].items()
        ]
        st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
        if task_key == "news":
            st.info("Hindi news classification is not currently supported or evaluated. Results for Hindi text are not a claim of multilingual accuracy.")


def _render_keywords(text: str) -> None:
    count = st.slider("Keywords to show", min_value=3, max_value=20, value=10, key="keyword_count")
    if st.button("Extract keywords", type="primary", key="run_keywords"):
        try:
            result = keywords.extract_keywords(text, count)
            save_analysis("Keyword Extraction", text, result, ", ".join(word for word, _ in result["tfidf"][:5]))
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Keyword Extraction")
    if result:
        tf_col, freq_col = st.columns(2)
        for column, title, values in (
            (tf_col, result["method"], result["tfidf"]),
            (freq_col, "Frequency ranked terms", result["frequency"]),
        ):
            with column:
                st.markdown(f"#### {title}")
                frame = pd.DataFrame(values, columns=["Keyword", "Score"])
                if not frame.empty:
                    st.dataframe(frame, width="stretch", hide_index=True)
                    chart = px.bar(frame.head(10), x="Score", y="Keyword", orientation="h", color="Score", color_continuous_scale=["#4bc7ec", "#9b7bff"])
                    chart.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, coloraxis_showscale=False)
                    st.plotly_chart(chart, width="stretch")
                else:
                    st.info("No keywords remain after filtering.")


def _render_summarization(text: str) -> None:
    count = st.slider("Summary sentence count", min_value=1, max_value=8, value=3, key="summary_length")
    if st.button("Summarize text", type="primary", key="run_summary"):
        try:
            result = summarization.summarize_text(text, count)
            result["original"] = text
            save_analysis("Text Summarization", text, result, result["summary"])
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Text Summarization")
    if result:
        original_size = len(result["original"])
        summary_size = len(result["summary"])
        reduction = 1 - summary_size / original_size if original_size else 0
        metric_col, reduction_col = st.columns(2)
        metric_col.metric("Sentences retained", len(result["sentences"]))
        reduction_col.metric("Character reduction", f"{reduction:.1%}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Original text")
            st.write(result["original"])
        with right:
            st.markdown("#### Extractive summary")
            st.info(result["summary"])
        st.caption(result["method"] + ". Summary sentences are selected from the original; no text is generated.")


def _render_language(text: str) -> None:
    if st.button("Detect language", type="primary", key="run_language"):
        try:
            result = language.detect_language(text)
            save_analysis("Language Detection", text, result, f"{result['name']} ({result['code'] or 'unknown'})")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Language Detection")
    if result:
        col1, col2 = st.columns(2)
        col1.metric("Language", result["name"])
        col2.metric("Language code", result["code"] or "—")
        if result["confidence"] is not None:
            st.progress(result["confidence"], text=f"Detection confidence: {result['confidence']:.1%}")
        if result.get("candidates"):
            candidates = pd.DataFrame(result["candidates"])
            candidates["language"] = candidates["code"].map(lambda code: language.LANGUAGE_NAMES.get(code, code.upper()))
            st.plotly_chart(
                px.bar(candidates, x="confidence", y="language", orientation="h", color="confidence", color_continuous_scale=["#4bc7ec", "#9b7bff"]),
                width="stretch",
            )
        st.caption(result["note"])


def _render_pos(text: str) -> None:
    if st.button("Tag parts of speech", type="primary", key="run_pos"):
        try:
            result = pos_tagging.tag_text(text)
            save_analysis("POS Tagging", text, result, f"{len(result['tags'])} tokens · {result['method']}")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("POS Tagging")
    if result:
        st.caption(result["method"])
        if result["tags"]:
            tag_frame = pd.DataFrame(result["tags"])
            table_col, chart_col = st.columns([1.4, 1])
            with table_col:
                st.dataframe(tag_frame, width="stretch", hide_index=True)
            with chart_col:
                counts = tag_frame["tag"].value_counts().rename_axis("POS tag").reset_index(name="Count")
                st.plotly_chart(px.bar(counts, x="POS tag", y="Count", color="POS tag"), width="stretch")
        else:
            st.info("No tokens were found.")


def _render_embeddings(text: str) -> None:
    st.markdown("Lexical BoW and TF-IDF vectors are built from each non-empty line.")
    st.caption("This module does not use pretrained or learned word embeddings.")
    left, right = st.columns(2)
    first = left.text_input("First word", placeholder="language", key="similarity_first")
    second = right.text_input("Second word", placeholder="text", key="similarity_second")
    if st.button("Create vectors", type="primary", key="run_embeddings"):
        try:
            vectors = embeddings.vectorize_text(text)
            similarity = embeddings.word_similarity(text, first, second) if first and second else None
            result = {
                "vectors": vectors,
                "similarity": similarity,
                "first_word": first,
                "second_word": second,
            }
            score = f" · word similarity {similarity['similarity']:.2f}" if similarity else ""
            save_analysis("Lexical Vectors", text, result, f"{len(vectors['bow_features'])} features{score}")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Lexical Vectors")
    if result:
        vectors = result["vectors"]
        doc_names = [f"Document {index + 1}" for index in range(len(vectors["bow"]))]
        bow, tfidf = st.tabs(["Bag of Words", "TF-IDF"])
        with bow:
            st.dataframe(pd.DataFrame(vectors["bow"], columns=vectors["bow_features"], index=doc_names), width="stretch")
        with tfidf:
            tfidf_frame = pd.DataFrame(vectors["tfidf"], columns=vectors["tfidf_features"], index=doc_names)
            st.dataframe(tfidf_frame, width="stretch")
            similarities = cosine_similarity(vectors["tfidf"])
            st.markdown("#### Document cosine similarity")
            st.plotly_chart(
                px.imshow(
                    similarities,
                    x=doc_names,
                    y=doc_names,
                    text_auto=".2f",
                    color_continuous_scale=["#151827", "#4bc7ec", "#9b7bff"],
                    zmin=0,
                    zmax=1,
                ),
                width="stretch",
            )
        st.caption("These are sparse lexical vectors, not pretrained semantic word embeddings.")
        if result["similarity"]:
            st.metric(
                f"Context similarity: {result['first_word']} ↔ {result['second_word']}",
                f"{result['similarity']['similarity']:.3f}",
            )
            st.caption(result["similarity"]["method"])
    st.button(
        "Load multi-sentence example",
        key="embedding_example",
        on_click=set_source_text,
        args=("Natural language processing studies language data.\nText analysis uses machine learning to discover patterns.",),
    )


def _render_topics(text: str) -> None:
    st.markdown("Enter one document per line. LDA discovers term patterns across this collection.")
    example_col, topics_col = st.columns([1, 1])
    example_col.button(
        "Load topic example",
        key="topic_example",
        on_click=set_source_text,
        args=(SAMPLE_DOCUMENTS,),
    )
    count = topics_col.slider("Number of topics", 1, 8, 3, key="topic_count")
    if st.button("Discover topics", type="primary", key="run_topics"):
        try:
            result = topic_modeling.model_topics(text.splitlines(), count)
            save_analysis("Topic Modeling", text, result, f"{len(result['topics'])} topics from {result['document_count']} documents")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Topic Modeling")
    if result:
        st.caption(f"LDA · {result['document_count']} documents · {result['feature_count']} terms")
        if result["topic_count"] != result["requested_topic_count"]:
            st.info(f"Using {result['topic_count']} topics: the request was capped by the available documents or vocabulary.")
        rows = []
        for item in result["topics"]:
            st.markdown(f"#### Topic {item['topic']}")
            st.write(" · ".join(word for word, _ in item["words"]))
            rows.extend({"Topic": f"Topic {item['topic']}", "Term": word, "Weight": weight} for word, weight in item["words"])
        if rows:
            fig = px.scatter(pd.DataFrame(rows), x="Weight", y="Term", color="Topic", size="Weight", color_discrete_sequence=["#9b7bff", "#4bc7ec", "#70d7b4", "#f0ac70", "#ef7195"])
            fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")


def _render_qa(context: str) -> None:
    question = st.text_input("Ask a question about this context", placeholder="What does the platform analyze?", key="qa_question")
    if st.button("Find answer", type="primary", key="run_qa"):
        try:
            result = question_answering.answer_question(context, question)
            save_analysis("Question Answering", context + "\nQuestion: " + question, result, result["answer"] or "No relevant answer found")
        except ValueError as error:
            st.warning(str(error))
    result = show_saved_result("Question Answering")
    if result:
        if result["supported"]:
            st.markdown("#### Most relevant sentence")
            st.info(result["answer"])
        else:
            st.warning("No sentence in the context matched the question strongly enough. Try using terms that appear in the passage.")
        st.progress(result["relevance"], text=f"Lexical relevance: {result['relevance']:.1%} (minimum {result['threshold']:.0%})")
        st.caption(result["method"] + ". A low relevance score means the context may not answer the question.")


def _render_chatbot() -> None:
    st.markdown(
        "A small FAQ and intent-matching demo for NLP concepts. "
        "It is intentionally not presented as a generative AI assistant."
    )
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about an NLP concept...")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        try:
            result = chatbot.respond(prompt)
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["chat_history"].append({"role": "assistant", "content": result["reply"]})
            record_analysis(st.session_state, "NLP Chatbot", prompt, result["intent"])
            st.rerun()
    if st.button("Clear chat history", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()
    render_downloads()


def render_page(page: str, on_navigate) -> None:
    if page == "Dashboard":
        render_dashboard(on_navigate)
        return
    description = next(item[2] for item in PAGES if item[1] == page)
    render_module_heading(page, description)
    if page == "NLP Chatbot":
        _render_chatbot()
        return
    if page == "Question Answering":
        st.caption("Paste a passage in the text workspace, then ask a question. The answer is retrieved from that passage.")
        _render_qa(render_text_workspace(SAMPLE_TEXT))
        return

    text = render_text_workspace(SAMPLE_TEXT)
    renderers = {
        "Text Preprocessing": _render_preprocessing,
        "Tokenization & N-Grams": _render_tokenization,
        "Text Statistics": _render_statistics,
        "Sentiment Analysis": _render_sentiment,
        "Named Entity Recognition": _render_ner,
        "Text Classification": _render_classification,
        "Keyword Extraction": _render_keywords,
        "Text Summarization": _render_summarization,
        "Language Detection": _render_language,
        "POS Tagging": _render_pos,
        "Lexical Vectors": _render_embeddings,
        "Topic Modeling": _render_topics,
    }
    renderer = renderers.get(page)
    if renderer is None:
        st.error(f"Unknown module: {page}")
    else:
        renderer(text)
