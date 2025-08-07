graph TD
    A[🎤 User Records Audio] --> B[📁 Audio Upload to Backend]
    B --> C[🔧 Audio Processing]
    C --> D[🎯 Whisper Transcription]
    D --> E[📊 Multi-Layer Analysis]
    E --> F[📝 Immediate Transcription Display]
    F --> G[🤖 AI Analysis Generation]
    G --> H[📋 Formatted Report]

    subgraph "Audio Processing Pipeline"
        C --> C1[Convert to WAV Format]
        C1 --> C2[Analyze Audio Properties]
        C2 --> C3[Extract Audio Features]
    end

    subgraph "Whisper Core Algorithm"
        D --> D1[Model: openai/whisper-base]
        D1 --> D2[Parameters: temperature=0.0, beam_size=5]
        D2 --> D3[Features: word_timestamps=True]
        D3 --> D4[Language: en, Task: transcribe]
        D4 --> D5[Extract Segments & Words]
        D5 --> D6[Calculate Word-Level Probabilities]
    end

    subgraph "Mathematical Analysis Engine"
        E --> E1[Confidence Analysis]
        E --> E2[Fluency Assessment]
        E --> E3[Pace Analysis]
        E --> E4[Pronunciation Clarity]
        E --> E5[Statistical Benchmarking]
        
        E1 --> F1[Average Confidence: sum of word probabilities divided by word count]
        E1 --> F2[Confidence Standard Deviation: sqrt of variance]
        E1 --> F3[Confidence Percentiles: P25, P50, P75, P90]
        
        E2 --> F4[Words Per Minute: total words times 60 divided by duration]
        E2 --> F5[Filler Rate: filler words divided by total words times 100]
        E2 --> F6[Fluency Score: weighted sum of confidence, pace, fillers, consistency]
        
        E3 --> F7[Pace Score: max of 1.0 or 5.0 minus absolute difference from 150 WPM]
        
        E4 --> F8[Word Confidence Score: average word confidence times 5]
        E4 --> F9[Consistency Score: max of 1.0 or 5.0 minus variance times 20]
        E4 --> F10[Recognition Score: recognized words divided by total words times 5]
        E4 --> F11[Complex Word Score: average complex word confidence times 5]
        E4 --> F12[Pronunciation Score: weighted sum of word confidence, consistency, recognition, complex]
        
        E5 --> F13[Performance Levels: beginner, intermediate, advanced, expert]
        E5 --> F14[Benchmark Scores: confidence 0.9, fluency 4.8, pace 4.5]
    end

    subgraph "4-Criterion Assessment System"
        H --> H1[📝 Content Analysis 1-5]
        H --> H2[🏗️ Structure Analysis 1-5]
        H --> H3[🎤 Clarity Analysis 1-5]
        H --> H4[⚡ Delivery Analysis 1-5]
        
        H1 --> I1[Accuracy of Information]
        H1 --> I2[Completeness of Response]
        H1 --> I3[Relevance to Question]
        
        H2 --> I4[Logical Flow]
        H2 --> I5[Organization]
        H2 --> I6[Coherence]
        
        H3 --> I7[Word Choice]
        H3 --> I8[Filler Word Usage]
        H3 --> I9[Pronunciation Clarity]
        H3 --> I10[Whisper Confidence Scores]
        
        H4 --> I11[Speaking Pace]
        H4 --> I12[Confidence Level]
        H4 --> I13[Energy Level]
        H4 --> I14[Volume Consistency]
    end

    subgraph "Performance Benchmarking"
        F13 --> J1[Beginner: confidence less than 0.5, fluency less than 2.5]
        F13 --> J2[Intermediate: confidence less than 0.7, fluency less than 3.5]
        F13 --> J3[Advanced: confidence less than 0.8, fluency less than 4.2]
        F13 --> J4[Expert: confidence greater than 0.9, fluency greater than 4.8]
    end

    subgraph "Weighting Systems"
        F6 --> K1[Fluency: 40 percent confidence plus 25 percent pace plus 20 percent fillers plus 15 percent consistency]
        F12 --> K2[Pronunciation: 40 percent word confidence plus 25 percent consistency plus 20 percent recognition plus 15 percent complex]
    end

    style A fill:#e1f5fe
    style H fill:#f3e5f5
    style F1 fill:#fff3e0
    style F6 fill:#fff3e0
    style F12 fill:#fff3e0
    style K1 fill:#e8f5e8
    style K2 fill:#e8f5e8 