Abstract

This project presents a modular, end-to-end Speech-to-Speech Translation System that converts spoken English input into spoken French output by integrating pretrained language models. The system bridges multiple complex tasks—Automatic Speech Recognition (ASR), Neural Machine Translation (NMT), and Text-to-Speech Synthesis (TTS)—using tools such as Google’s SpeechRecognition API, Hugging Face’s MarianMT model (opus-mt-en-fr), and gTTS for speech synthesis. By eliminating the need for manual data collection or model training, the approach emphasizes rapid development, scalability, and accessibility.

A curated dataset of 20 English sentences was synthesized into audio and processed through the pipeline. The system’s performance was evaluated using metrics such as Word Error Rate (WER), BLEU, METEOR, and Character Error Rate (CER), as well as a custom Overall Quality Score. Results demonstrated robust performance under general conditions, achieving a 100% processing success rate and an average BLEU score of ~0.64.

The project highlights both the potential and current limitations of modular, pretrained architectures for multilingual speech processing. Its contributions lie not only in demonstrating a functional pipeline, but also in laying a foundation for future enhancements, including multilingual, real-time, or emotion-aware translation systems aimed at increasing global accessibility and cross-cultural communication.

1.Problem Statement 
Title:Speech-to-Speech Translation System Using Pre Trained Language Models

Language is one of the most fundamental tools for human communication, yet it remains one of the biggest barriers in global interaction. With globalization on the rise, cross-linguistic communication has become a necessity in education, healthcare, travel, commerce, and international diplomacy. While modern tools such as Google Translate and DeepL have made considerable progress in bridging this gap, most existing solutions are limited to text-based interfaces. For individuals who are illiterate, visually impaired, or in time-critical situations, the need for seamless speech-to-speech translation becomes even more crucial.

This project seeks to address this need by developing a Speech-to-Speech Translation System that takes spoken English as input, translates it into spoken French, and delivers the output through audio. This pipeline combines three complex sub-tasks: Automatic Speech Recognition (ASR), Neural Machine Translation (NMT), and Text-to-Speech Synthesis (TTS). Each of these components introduces unique challenges that must be addressed with efficient, accurate, and scalable solutions.

The problem of direct speech-to-speech translation is inherently complex due to the loss of semantic information that may occur at each stage. For example, the ASR component must deal with variations in speaker accent, pace, noise interference, and pronunciation. Even a slight transcription error can propagate through the translation model, leading to significantly different or incorrect output. Furthermore, the translation component must handle syntactic, semantic, and cultural nuances between languages. Finally, the TTS engine must synthesize natural-sounding speech that preserves meaning and fluency.

Instead of building models from scratch, this project leverages pretrained models known for their performance and generalizability. These include Google's SpeechRecognition API for ASR, Hugging Face's MarianMT (Helsinki-NLP/opus-mt-en-fr) for English-to-French translation, and Google Text-to-Speech (gTTS) for French audio synthesis. The system is designed to support both interactive mode using microphone input and batch mode using uploaded audio files.

This project also focuses on objective evaluation of the complete pipeline, using metrics such as Word Error Rate (WER), BLEU score, METEOR score, and Character Error Rate (CER). These metrics help to quantify both the accuracy of transcription and the quality of the translation, providing an empirical basis for improvement.The ultimate goal is to provide a modular, reusable, and user-friendly pipeline that performs high-quality English-to-French speech translation and can serve as a foundation for future multilingual, bidirectional, or real-time speech translation systems


2.Methodology
This project proposes a comprehensive speech-to-speech translation system designed to convert spoken English into spoken French. The pipeline is designed as a modular, multi-stage workflow that encapsulates various key components in natural language processing and speech technologies. These include: English text-to-speech (TTS) synthesis, automatic speech recognition (ASR), neural machine translation (NMT), French speech synthesis, interactive interface integration, evaluation using standardized metrics, and data visualization. The modular nature ensures that each stage can be developed, tested, and optimized independently while contributing to the overall system accuracy.
The core motivation behind this methodology is to simulate a real-world application of multilingual speech translation systems, which are increasingly critical in global communication, real-time conferencing, international customer service, and accessibility technologies. Instead of relying on real-time user inputs initially, the project uses synthetically generated English speech to maintain consistency, reduce noise, and control variability in the dataset. This ensures a more reliable foundation for testing the core pipeline components.
A major emphasis is placed on automation and reproducibility. From converting English text into speech, transcribing it, translating it to French, and converting it back into French audio, each step is handled programmatically with minimal human intervention. This end-to-end automation is especially beneficial for scalability and for deploying similar systems in multiple language pairs.
Moreover, state-of-the-art pre-trained models and libraries are leveraged at each stage. These include Google's TTS and Speech Recognition APIs, Hugging Face's MarianMT transformer model, and Gradio for user interface creation. These choices strike a balance between development ease and performance, ensuring that the system is both accessible and powerful.
To ensure that the system performs effectively, an array of evaluation metrics is incorporated. Metrics such as Word Error Rate (WER) and BLEU Score allow objective measurement of transcription and translation quality, while additional metrics like Character Error Rate (CER) and METEOR help in understanding linguistic and semantic accuracy. The system also generates visual analytics, using libraries such as matplotlib, to compare the pipeline’s performance across test samples.
In summary, this methodology is built to reflect real-world constraints while utilizing modern NLP and speech technologies. It serves not only as a demonstration of speech-to-speech translation capabilities but also as a flexible foundation for further research and industrial applications.

2.1. English Text-to-Speech Synthesis
To simulate consistent and clean English speech inputs, the system begins with synthetic speech generation from predefined English sentences. Instead of using human-recorded audio—which may include variations in tone, pace, and noise—the Google Text-to-Speech (gTTS) library is used to convert English text into speech. These audio files are initially saved in .mp3 format and are then converted to .wav format using the pydub library. .wav is the preferred format for speech recognition tasks due to its lossless, uncompressed nature.
Key Highlights:
Uses gTTS to generate clear English speech samples.
.mp3 converted to .wav for compatibility with recognition models.
 Eliminates variability caused by human voices.
2.2. Automatic Speech Recognition (ASR)
The synthesized English audio is fed into an automatic speech recognition (ASR) system to extract the spoken text. This module uses Python’s SpeechRecognition library with the Google Web Speech API as the backend. The recognizer loads the .wav file and attempts to transcribe the audio into English text. Exception handling is included to manage errors such as unrecognized speech or failed API calls. ASR accuracy is vital because translation quality directly depends on the transcription.
Key Highlights:
Leverages Google Web Speech API for transcription accuracy.
Implements error handling (UnknownValueError, RequestError) for robustness.
Converts .wav audio to text efficiently for downstream processing.
2.3. Neural Machine Translation (NMT)
Once the English text is available, it is translated into French using a neural machine translation model. This project employs the Helsinki-NLP/opus-mt-en-fr model available on Hugging Face, based on the MarianMT architecture. The English sentences are tokenized and passed through the model to generate corresponding French sentences. The model ensures contextual accuracy and preserves meaning, which is critical in spoken communication.
Key Highlights:
Uses a pre-trained MarianMT model for reliable translation.
Tokenizes and decodes using Hugging Face Transformers.
Maintains semantic and grammatical correctness.

2.4. French Text-to-Speech Synthesis
After translating the English text into French, the system uses gTTS again—this time with the language set to 'fr'—to generate spoken French audio. The output is saved as an .mp3 file and played back using the playsound library. Temporary files are removed after playback to maintain performance and cleanliness.

Key Highlights:

Converts French text into lifelike speech using gTTS.
Supports real-time playback of French audio.
Performs clean up of temporary audio files.

2.5. Real-Time Interface with Gradio
To offer an accessible and interactive user experience, a front-end interface is created using the Gradio library. The interface supports both pre-recorded uploads and real-time microphone input. The audio is passed through the full pipeline—ASR, NMT, and TTS—and returns the recognized English text, the French translation, and the spoken French output. Gradio simplifies deployment and testing, making the pipeline accessible even to non-technical users.
Key Highlights:
User-friendly Gradio interface for audio input and playback.
Supports both file upload and microphone recording.
Displays intermediate outputs for transparency.
2.6. Dataset Creation and Processing
A curated dataset of 20 English sentences is manually prepared to ensure control over inputs and outputs. Each sentence is converted to .mp3 and then to .wav using the earlier TTS method. This controlled dataset provides a standardized way to evaluate the pipeline’s accuracy and behavior. Since the dataset is generated synthetically, it eliminates environmental noise, speaker variation, and other inconsistencies.
Key Highlights:
Dataset contains 20 well-formed English sentences.
Synthesized uniformly using gTTS to avoid noise.
Enables repeatable and measurable evaluations.
2.7. Evaluation and Performance Metrics
To assess the quality of both ASR and translation, several evaluation metrics are used:
ASR Evaluation Metrics:
Word Error Rate (WER): Measures the total number of insertions, deletions, and substitutions needed to match predicted text to the reference.
Match Error Rate (MER): Gives an overview of overall inaccuracy.
Word Information Lost (WIL): Measures the proportion of lost information in the recognized output.
NMT Evaluation Metrics:
BLEU Score: Evaluates n-gram overlaps between machine output and reference translation.
METEOR Score: Considers semantic similarity, synonym matching, and word order.
Character Error Rate (CER): Compares character-level accuracy between translations.
Additionally, a custom performance metric is proposed:
Overall Quality Score = (1 – WER) × BLEU Score
This combines the ASR and NMT accuracy into a single value for easier comparative analysis.
Key Highlights:
Standard ASR metrics: WER, MER, WIL.
NMT metrics: BLEU, METEOR, CER.
Composite metric for overall quality tracking.
2.8. Visualization of Results
The system includes a visualization module using matplotlib to graphically present evaluation results. This helps to quickly identify underperforming samples or patterns in translation errors. Metrics like WER, BLEU, and Overall Quality Score are plotted across all test samples.
Key Highlights:
Bar and line plots generated using matplotlib.
Visualizations offer insights into system strengths and weaknesses.
Helps validate improvements during iterative development.
2.9. Pipeline Automation and Modularity
The entire system is modular, with each task (TTS, ASR, NMT, evaluation) implemented as a standalone function. This allows for easy updates, testing, and debugging. Temporary files are automatically managed, and exception handling ensures the system is robust even when individual modules fail.
Key Highlights:
Modular functions for clean, reusable code.
Automatically deletes temporary files to save resources.
Flexible for upgrades (e.g., replacing ASR with Whisper or TTS with Tacotron).

3.Implementation
Overview
The implementation stage translates the above methodologies into an operational system using Python and various open-source tools and APIs. Emphasis was placed on modularity, code readability, and efficient data flow between components.
3.1. English Audio Synthesis Implementation
The English sentences were programmatically converted into speech using the gTTS library.
Code :
	from gtts import gTTS
def synthesize_english_text(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
The generated audio files were standardized to 16 kHz sampling rate to optimize ASR performance.
Batch processing was implemented to generate audio for all dataset sentences.
3.2. Speech Recognition Integration
Using the SpeechRecognition library, English audio was transcribed:
Implementation Details:
Audio was loaded using AudioFile class.
Google's Web Speech API was called for recognition.
Code :
	import speech_recognition as sr
recognizer = sr.Recognizer()
def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    return text
Error handling was incorporated to manage API timeouts and unclear audio.
3.3. Neural Machine Translation
The MarianMT model was loaded using Hugging Face Transformers.
Steps:
Initialize tokenizer and model.
Tokenize input English text.
Generate French translation tokens.
Decode tokens to readable text
Code :
	from transformers import MarianMTModel, MarianTokenizer
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text
Batch translation support allowed processing multiple sentences efficiently.
3.4. French Audio Generation
The translated text was converted to French speech using gTTS:

Similar to English synthesis but with lang='fr'.
Audio files were generated and stored for playback.
	

Code :
def synthesize_french_text(text, filename='french_audio.mp3'):
    tts = gTTS(text=text, lang='fr')
    tts.save(filename)
    print(f"French audio saved to {filename}")
3.5. Gradio Interface Construction
A multi-component Gradio interface was developed to orchestrate the full pipeline:
Input text box for English sentences.
Buttons to generate English audio, transcribe, translate, and synthesize French speech.
Playback widgets for both English and French audio.
Display areas for text and metric outputs.
Sample Gradio Code:
	import gradio as gr
def full_pipeline(text):
    synthesize_english_text(text, 'input.wav')
    transcription = transcribe_audio('input.wav')
    translation = translate_text(transcription)
    synthesize_french_text(translation, 'output.wav')
    return transcription, translation, 'input.wav', 'output.wav'

iface = gr.Interface(fn=full_pipeline,
                     inputs=gr.Textbox(label="English Text"),
                     outputs=[gr.Textbox(label="Transcription"),
                              gr.Textbox(label="Translation"),
                              gr.Audio(label="English Audio"),
                              gr.Audio(label="French Audio")])
iface.launch()
3.6. Dataset Handling and Automation
Scripts were developed to:
Load dataset from CSV.
Iterate through each sentence.
Run full pipeline automatically.
Store outputs and evaluation metrics for further analysis.
Code :
import pandas as pd
def process_dataset(csv_path):
    data = pd.read_csv(csv_path)
    results = []
    for idx, row in data.iterrows():
        english_text = row['english_text']
        print(f"Processing sentence {idx+1}: {english_text}")
        synthesize_english_text(english_text, f'input_{idx}.mp3')
        transcription = transcribe_audio(f'input_{idx}.mp3')
        translation = translate_text(transcription)
        synthesize_french_text(translation, f'output_{idx}.mp3')
        results.append({
            'english_text': english_text,
            'transcription': transcription,
            'translation': translation,
            'reference': row.get('french_reference', '')
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('translation_results.csv', index=False)
Automation minimized manual intervention and supported batch evaluation.
3.7. Evaluation Metrics Computation
After translation, outputs were compared with references using metric libraries:
WER and CER computed on transcription accuracy.
BLEU and METEOR scores computed for translation quality.
Results saved in CSV format for reporting.
Code:
from jiwer import wer
error = wer(reference_text, hypothesis_text)
3.8. Visualization Scripts
Data visualization was implemented using:
Pandas for data manipulation.
Matplotlib/Seaborn for plotting.
Scripts generate comparative charts and tables for reporting results.
Code :
import matplotlib.pyplot as plt
import seaborn as sns
def plot_metric_scores(results_csv):
    df = pd.read_csv(results_csv)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x=df.index, y='BLEU', color='skyblue')
    plt.title('BLEU Scores per Sentence')
    plt.xlabel('Sentence Index')
    plt.ylabel('BLEU Score')
    plt.show()
3.9. Error Handling and Optimization
Throughout implementation:
Exceptions were caught to handle network errors, API limits, and unexpected inputs.
Audio preprocessing steps like noise reduction and normalization were tested to improve ASR accuracy.
Parameter tuning (e.g., beam search width in translation) optimized output quality.





4.Results and Evaluation
Overview 
The purpose of this section is to analyze the performance of the proposed English-to-French Speech-to-Speech Translation System across three main tasks:

Automatic Speech Recognition (ASR)
Machine Translation (MT)
Text-to-Speech Synthesis (TTS)

To evaluate the system’s effectiveness, we designed an experimental pipeline comprising 20 manually curated English sentences. These were converted into speech using a TTS engine, transcribed using ASR, translated into French using a MarianMT model, and then converted into French speech using gTTS.A variety of metrics were used to assess each stage of the pipeline, including Word Error Rate (WER), BLEU, METEOR, Character Error Rate (CER), and a composite Overall Quality Score.
4.1. Experimental Setup
4.1.1 Dataset Preparation
A set of 20 diverse English sentences was created covering different domains such as:

Daily conversations
Navigation queries
Technical statements
News-like sentences
Narrative structures
These were chosen to test how well the system generalizes across speech styles and vocabulary.
4.1.2 System Workflow
The steps followed for each sentence were:

Text-to-Speech (TTS) in English using gTTS
Speech Recognition of generated audio using SpeechRecognition (Google ASR)
Translation of recognized English to French using MarianMT (opus-mt-en-fr)
French TTS of translated sentence using gTTS

4.2. Evaluation Metrics

Metric 
Description
WER
Measures transcription error by comparing reference and hypothesis word by word.
MER
Match Error Rate — counts correct words and mismatches.
WIL
Word Information Lost — a metric that penalizes over-insertion or deletion of words.
BLEU
Evaluates translation accuracy based on n-gram overlap with the reference.
METEOR
Semantic metric considering stemming, synonymy, and word order.
CER	
Measures edit distance at the character level.
Overall Quality Score	Custom metric: 
(1−WER)×BLEU



4.3. Quantitative Results
4.3.1 Aggregate Metrics
Metric
Mean ± Std. Dev
WER
0.2328 ± 0.1834
MER	
0.2328 ± 0.1834
WIL
0.3774 ± 0.1919
BLEU	
0.6423 ± 0.2456
METEOR	
0.8119 ± 0.2056	
CER
0.1184 ± 0.1714
Overall Score
0.5266 ± 0.2203

Success Rate	100%
4.3.2 Interpretation
The Word Error Rate (WER) of 23% suggests the ASR was reasonably accurate, with most errors being due to filler words or misrecognized names.
A BLEU score of 0.64 indicates strong translation quality, though lower on grammatically complex sentences.
METEOR and CER reinforce that the translated output retains both word-level and character-level fidelity.
Overall pipeline quality is rated as Fair, which aligns with expectations for a non-finetuned, general-purpose model chain.
4.4 Qualitative Results
4.4.1 Sample Outputs
Example 
Original English: “Could you please tell me how to get to the nearest train station?”
Recognized ASR: “could you please tell me how to get to the nearest train station”
Ground Truth French: “Pouvez-vous me dire comment vous rendre à la gare la plus proche ?”
Translated French: “Pouvez-vous me dire comment arriver à la gare la plus proche ?”
BLEU: 0.6761
Overall Quality: 0.5721

Example 
Original: “We should plant more trees to help combat climate change.”
Recognized ASR: “climate change”
Translation: “changements climatiques”
BLEU: 0.0000
Overall Quality: 0.0000
Observation: ASR failed; translation still made sense for the short phrase
4.5. Visual Results
4.5.1 Speech Recognition Error Trends
Description:
The line graph plots the three major ASR error metrics — Word Error Rate (WER), Match Error Rate (MER), and Word Information Lost (WIL) — for all 20 input examples. Most examples have WER values between 0.1 and 0.3, with only a few outliers (e.g., example 20) exhibiting higher error due to poor transcription or audio artifacts.
Observation:
Example 20 had WER > 0.9, showing a severe recognition failure.
The ASR system was generally robust with an average WER of ~23%.


4.5.2 Translation Scores
Description:
This plot compares the BLEU and METEOR scores for each example, reflecting how well the translated French text aligns with the ground truth reference.
Observation:
Most BLEU scores are above 0.6, indicating strong n-gram overlap.
METEOR scores are consistently higher, showing the system performs well in capturing semantic meaning, not just token-level matches.
The system translated simple declarative sentences better than interrogatives or complex statements.

4.5.3 Character-Level Error Analysis
Description:
The CER plot measures the character-level difference between the translated French output and the reference. Lower values mean higher fidelity.
Observation:
CER remains low across most examples (< 0.2), showing good accuracy.
Spikes in CER correspond with examples where ASR failed partially, affecting thedownstream translation.
4.5.4 Composite Pipeline Quality
Description:
This bar graph shows the overall quality score for each example, calculated as:
Score=(1−WER)×BLEU
Observation:
Most examples fall between 0.4 and 0.7, indicating fair to good end-to-end pipeline quality.
Example 20 shows 0 quality, due to ASR failure.
The red line represents the average score, useful for benchmarking future models.

4.6. Error Analysis
4.6.1 Causes of ASR Errors
Fast or unclear TTS pronunciation
ASR bias toward short utterances
Punctuation ignored or dropped
4.6.2 Translation Weaknesses
Slight variation in sentence structures
Missing idiomatic conversions
No fine-tuning for speech domain
4.6.3 Synthesis Challenges
Minor robotic tone in gTTS output
No emphasis or emotion carried forward






Conclusion

The development of the Speech-to-Speech Translation System presented in this project is a significant step toward building accessible, multilingual communication tools. The project aimed to design a modular pipeline capable of translating spoken English input into spoken French output using only pretrained models, without requiring large-scale data collection or model fine-tuning.

This system successfully integrates three key components of modern AI-driven language processing: Automatic Speech Recognition (ASR), Neural Machine Translation (NMT), and Text-to-Speech Synthesis (TTS). Using tools such as Google’s SpeechRecognition API, Hugging Face's MarianMT model (opus-mt-en-fr), and gTTS for speech synthesis, the project created a functional and testable end-to-end system. The design emphasizes usability by supporting both real-time interaction via microphone input and batch-mode processing via uploaded files.

The system was evaluated using 20 carefully selected English sentences, allowing for a comprehensive examination of its effectiveness. Quantitative metrics such as WER (Word Error Rate), BLEU score, METEOR score, and CER (Character Error Rate) were used to assess each component of the pipeline. The results showed that while the system performs reliably in general scenarios—with an average WER of ~23% and a BLEU score of ~64%—it does face challenges in handling nuanced language structures, speaker variability, and longer sentences.
A key contribution of this project is the custom composite metric combining transcription and translation quality into an Overall Quality Score, which helped in assessing the pipeline as a whole. The success rate of 100% in producing output (even if imperfect in some cases) demonstrates the robustness of the design under diverse inputs.

However, several limitations were observed. The ASR component struggled with background noise or fast-paced speech, and the translation model occasionally produced literal translations without context adaptation. The TTS engine, while functional, lacked expressiveness or emotional tone, which is essential for natural conversation. These limitations provide meaningful directions for future improvements.

Looking ahead, this system can be significantly enhanced by integrating state-of-the-art open-source ASR models like Whisper, adding support for multiple language pairs, and incorporating emotion-aware TTS engines. The potential applications of such systems are vast—from tourism and healthcare to education and emergency services—making it a valuable area of continued research and innovation.

In conclusion, this project demonstrates that a reliable and modular speech-to-speech translation system can be developed using publicly available resources. It lays a strong foundation for more sophisticated multilingual communication tools and contributes to the broader vision of inclusive, AI-driven language accessibility.

