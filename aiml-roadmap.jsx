import { useState, useEffect } from "react";

const WEEKS = [
  {
    id: 1,
    title: "Python & Data Foundations",
    subtitle: "WEEK 01",
    accent: "#00FFB2",
    bg: "rgba(0,255,178,0.06)",
    days: [
      {
        day: 1,
        title: "Python Speed Run",
        topic: "Python Fundamentals",
        emoji: "🐍",
        theory: ["Variables, loops, functions, list comprehensions", "NumPy arrays & vectorized operations", "Matplotlib crash course"],
        project: {
          name: "Spotify Mood Analyzer",
          icon: "🎵",
          desc: "Build a CLI tool that reads a CSV of song data (BPM, energy, valence, danceability). It crunches the numbers and prints a savage personality roast: 'You listen to 78% high-energy music — classic caffeine-addicted overachiever energy.'",
          skills: ["Python lists", "CSV file I/O", "Basic math", "f-strings", "CLI design"],
          time: "2–3 hrs",
          difficulty: "BEGINNER",
          dataset: "Kaggle Spotify Dataset"
        },
        learn: ["How Python handles data types", "Why list comprehensions are faster", "How to structure a CLI tool"]
      },
      {
        day: 2,
        title: "NumPy Ninja",
        topic: "Array Computing",
        emoji: "⚡",
        theory: ["Array broadcasting & vectorization", "Shape manipulation & slicing", "Linear algebra operations"],
        project: {
          name: "Heatwave Streak Detector",
          icon: "🌡️",
          desc: "Load 10 years of real temperature data (NOAA open data). Use NumPy boolean indexing to detect heatwave streaks (5+ consecutive days above 40°C). Plot frequency by decade. Watch climate change reveal itself in your code.",
          skills: ["NumPy boolean indexing", "Broadcasting", "Rolling windows", "Matplotlib plotting"],
          time: "3 hrs",
          difficulty: "BEGINNER",
          dataset: "NOAA Climate Data (free)"
        },
        learn: ["Why vectorization beats for-loops by 100x", "How to think in arrays", "Real climate data handling"]
      },
      {
        day: 3,
        title: "Pandas Powerhouse",
        topic: "Data Manipulation",
        emoji: "🐼",
        theory: ["DataFrames, groupby, merge, pivot", "Handling missing data strategies", "Method chaining for clean code"],
        project: {
          name: "IPL Match Oracle",
          icon: "🏏",
          desc: "Use the full IPL dataset (2008–2023). Build a pandas query engine that answers: 'Who wins when MI tosses first at Wankhede on a D/N match in April?' Chain groupby operations to reveal hidden patterns. Find the team that chokes the most in finals.",
          skills: ["Pandas groupby", "Multi-index", "Boolean filtering", "Data cleaning"],
          time: "3–4 hrs",
          difficulty: "BEGINNER",
          dataset: "Kaggle IPL Dataset"
        },
        learn: ["Data cleaning in the real world", "How groupby actually works", "Sports analytics basics"]
      },
      {
        day: 4,
        title: "Visualize Like a Designer",
        topic: "Data Visualization",
        emoji: "🎨",
        theory: ["Matplotlib subplot grids", "Seaborn statistical plots", "Color theory & storytelling with data"],
        project: {
          name: "World Happiness Atlas",
          icon: "🌍",
          desc: "Use UN World Happiness Report data. Create a 6-panel visual dashboard: scatter plots, bar races, correlation heatmaps, violin plots. Answer: 'Does GDP really buy happiness?' Tell the whole story in charts alone — no text needed.",
          skills: ["Matplotlib subplots", "Seaborn", "Annotations", "Color palettes"],
          time: "3 hrs",
          difficulty: "BEGINNER",
          dataset: "World Happiness Report (Kaggle)"
        },
        learn: ["When to use which chart type", "How design choices mislead or enlighten", "Annotation as storytelling"]
      },
      {
        day: 5,
        title: "Stats That Actually Matter",
        topic: "Probability & Statistics",
        emoji: "🎲",
        theory: ["Distributions, mean, variance, CLT", "Hypothesis testing & p-values", "Correlation vs causation (critical!)"],
        project: {
          name: "Casino Fairness Investigator",
          icon: "🎰",
          desc: "Simulate 100,000 roulette spins. Use chi-square tests to 'prove' whether a casino's wheel is rigged. Generate a statistical verdict report with confidence intervals, p-values, and a judge's ruling. Spoiler: the house always wins.",
          skills: ["SciPy stats", "Monte Carlo simulation", "Hypothesis testing", "Confidence intervals"],
          time: "3 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "Self-generated (simulation)"
        },
        learn: ["What p-values actually mean", "Why correlation lies", "Monte Carlo simulation power"]
      },
      {
        day: 6,
        title: "Linear Algebra Unlocked",
        topic: "Math for ML",
        emoji: "🧮",
        theory: ["Vectors, matrices, dot products", "Eigenvalues & eigenvectors intuitively", "Why this is the backbone of every ML algorithm"],
        project: {
          name: "Image Compression Machine",
          icon: "🖼️",
          desc: "Take any high-res photo. Use Singular Value Decomposition (numpy.linalg.svd) to compress it to 5%, 10%, 25%, 50% quality. Show animated side-by-side comparisons. This IS the math behind JPEG, Netflix streaming, and neural networks.",
          skills: ["NumPy linalg", "SVD", "Image as matrix", "Subplot animation"],
          time: "2–3 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "Any image you own"
        },
        learn: ["Why SVD is everywhere in ML", "How images are just numbers", "The beauty of matrix decomposition"]
      },
      {
        day: 7,
        title: "Week 1 Boss Battle",
        topic: "Full EDA Pipeline",
        emoji: "⚔️",
        theory: ["EDA best practices", "How to ask good questions from data", "Presenting insights as a story"],
        project: {
          name: "Mumbai Traffic Chaos Report",
          icon: "🚗",
          desc: "Use Mumbai accident data from data.gov.in. Full pipeline: load → clean → analyze → visualize → insights. Build 8+ charts. Find: which junction kills the most, which hour is most dangerous, which vehicle type is highest risk. Write it like a data journalist exposé.",
          skills: ["Full pandas + seaborn pipeline", "EDA methodology", "Data journalism"],
          time: "5–6 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "data.gov.in accident records"
        },
        learn: ["How to structure an entire data project", "Real messy data vs clean datasets", "Data storytelling"]
      }
    ]
  },
  {
    id: 2,
    title: "Classical Machine Learning",
    subtitle: "WEEK 02",
    accent: "#FF6B6B",
    bg: "rgba(255,107,107,0.06)",
    days: [
      {
        day: 8,
        title: "The ML Mindset",
        topic: "Foundations of ML",
        emoji: "🧠",
        theory: ["Train/val/test splits & why they matter", "Overfitting vs underfitting visually", "Bias-variance tradeoff intuitively"],
        project: {
          name: "Pizza Price Predictor (Scratch)",
          icon: "🍕",
          desc: "Create a 500-row synthetic pizza dataset (size, toppings, restaurant tier, city tier, delivery distance). Implement linear regression FROM SCRATCH using NumPy — no sklearn. Write gradient descent yourself. Feel every weight update. Then compare with sklearn's answer.",
          skills: ["Gradient descent from scratch", "Loss function math", "NumPy only", "Validation curves"],
          time: "4 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "Self-generated synthetic data"
        },
        learn: ["What ML is actually doing mathematically", "Why gradient descent works", "The difference between ML and statistics"]
      },
      {
        day: 9,
        title: "Sklearn Enters the Chat",
        topic: "Regression & Classification",
        emoji: "📊",
        theory: ["Sklearn API patterns (fit/predict/transform)", "Regularization: Ridge vs Lasso", "ROC curves, AUC, confusion matrix"],
        project: {
          name: "Hospital Readmission Risk",
          icon: "🏥",
          desc: "UCI Diabetes 130-hospitals dataset. Predict 30-day readmission. The twist: tune classification threshold to minimize false negatives (missing a sick patient costs a life, not just accuracy). Write a clinical recommendation report. This is real medical AI stakes.",
          skills: ["Logistic regression", "Threshold tuning", "Precision/recall tradeoff", "Clinical metrics"],
          time: "4 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "UCI ML Repository"
        },
        learn: ["Why accuracy is a useless metric alone", "How threshold tuning changes everything", "Real-world ML consequences"]
      },
      {
        day: 10,
        title: "Trees Are Powerful",
        topic: "Decision Trees & Random Forest",
        emoji: "🌲",
        theory: ["Information gain & Gini impurity", "Random forest bagging intuition", "Feature importance & selection"],
        project: {
          name: "Crop Yield Oracle",
          icon: "🌾",
          desc: "Indian agriculture dataset (data.gov.in). Predict crop yield from rainfall, fertilizer type, soil quality, season, and state. Visualize the full decision tree. Extract what matters most for each crop. Build an interactive CLI: 'Enter your farm details → get yield prediction and top 3 factors.'",
          skills: ["Decision Trees", "Random Forest", "Feature importance", "Tree visualization"],
          time: "3–4 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "data.gov.in agriculture data"
        },
        learn: ["How trees actually split data", "Why ensemble > single model", "Feature importance as domain insight"]
      },
      {
        day: 11,
        title: "Boost Everything",
        topic: "XGBoost & Gradient Boosting",
        emoji: "🚀",
        theory: ["Boosting vs bagging: the key difference", "XGBoost hyperparameters that matter", "Early stopping to prevent overfit"],
        project: {
          name: "Bengaluru Flat Flipper",
          icon: "🏠",
          desc: "Bengaluru house prices dataset. XGBoost to predict price per sqft. Grid search optimal params. Build a CLI house price estimator: enter BHK, area sqft, location, age of building → get predicted price + ±10% confidence range. Compare to actual listings. Beat random forest.",
          skills: ["XGBoost", "GridSearchCV", "Hyperparameter tuning", "Feature engineering"],
          time: "4 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "Kaggle Bengaluru House Prices"
        },
        learn: ["Why XGBoost wins Kaggle competitions", "What hyperparameters actually control", "Model comparison methodology"]
      },
      {
        day: 12,
        title: "Find Hidden Tribes",
        topic: "Unsupervised Learning",
        emoji: "🔍",
        theory: ["K-Means algorithm step by step", "DBSCAN for anomaly detection", "Elbow method & silhouette score"],
        project: {
          name: "Customer Tribe Finder",
          icon: "🛍️",
          desc: "Retail transaction dataset. K-Means to find customer segments. The fun part: NAME each cluster based on behavior. 'The Bargain Hunters', 'Loyal Whales', 'Weekend Warriors', 'Ghost Subscribers'. Build a visualization of customer universe. Present it as a marketing strategy deck.",
          skills: ["KMeans", "PCA for 2D viz", "Cluster interpretation", "Business translation"],
          time: "3–4 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "Kaggle Mall Customer / Online Retail"
        },
        learn: ["When you don't have labels", "How business translates ML clusters", "PCA for visualization"]
      },
      {
        day: 13,
        title: "Reduce the Noise",
        topic: "Dimensionality Reduction",
        emoji: "🌀",
        theory: ["PCA: what it really does geometrically", "t-SNE for high-dimensional visualization", "When to use PCA vs t-SNE vs UMAP"],
        project: {
          name: "Cancer Cell Constellation",
          icon: "🧬",
          desc: "Wisconsin Breast Cancer dataset (30 features). Use PCA to collapse to 3D, then t-SNE to 2D. Visualize how malignant vs benign cells form distinct galaxies in feature space. Overlay cluster boundaries. The math is literally saving lives.",
          skills: ["PCA", "t-SNE", "3D scatter plots", "Sklearn Pipeline"],
          time: "3 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "UCI Wisconsin Breast Cancer"
        },
        learn: ["The curse of dimensionality", "Why t-SNE clusters matter", "How to visualize invisible dimensions"]
      },
      {
        day: 14,
        title: "Week 2 Boss Battle",
        topic: "End-to-End ML Pipeline",
        emoji: "⚔️",
        theory: ["Sklearn Pipelines for production", "Feature engineering at scale", "Model selection & fair comparison"],
        project: {
          name: "Indian Flight Delay Predictor",
          icon: "✈️",
          desc: "DGCA Indian flight data. Full pipeline: feature engineering (extract hour, day, month, season from datetime; encode airlines, airports; create delay history features). Compare 5 models fairly using the same pipeline. Build a CLI: 'Enter your flight details → probability of delay + how late.' Beat the airline's own estimates.",
          skills: ["Full sklearn Pipeline", "Feature engineering", "5-model comparison", "Production-ready code"],
          time: "6 hrs",
          difficulty: "ADVANCED",
          dataset: "DGCA / Kaggle Indian Flights"
        },
        learn: ["How real ML projects are structured", "Feature engineering as the real skill", "Pipeline prevents data leakage"]
      }
    ]
  },
  {
    id: 3,
    title: "Deep Learning & Neural Networks",
    subtitle: "WEEK 03",
    accent: "#A78BFA",
    bg: "rgba(167,139,250,0.06)",
    days: [
      {
        day: 15,
        title: "Neurons Fire",
        topic: "Neural Network Fundamentals",
        emoji: "⚡",
        theory: ["Perceptron → MLP architecture", "Backpropagation: the math behind learning", "Activation functions & their roles"],
        project: {
          name: "Digit Whisperer (Two Ways)",
          icon: "🔢",
          desc: "MNIST, but actually interesting. First: build a 3-layer neural network ENTIRELY from scratch with NumPy — write forward pass, loss, backward pass, weight updates yourself. Achieve 95%+ accuracy. Then build identical network in Keras in 10 lines. Compare. Visualize what neurons in each layer 'see'.",
          skills: ["NumPy NN from scratch", "Backprop math", "Keras Sequential", "Activation visualization"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "MNIST (Keras built-in)"
        },
        learn: ["What neural networks actually compute", "Why backprop is the greatest algorithm", "The abstraction Keras provides"]
      },
      {
        day: 16,
        title: "CNNs See the World",
        topic: "Convolutional Neural Networks",
        emoji: "👁️",
        theory: ["Convolution, pooling, stride — visually", "CNN architecture design principles", "Data augmentation strategies"],
        project: {
          name: "Plant Disease Doctor",
          icon: "🌿",
          desc: "PlantVillage dataset: 38 plant diseases from leaf photos. Build a CNN that helps farmers diagnose diseases with their phone. The challenge: visualize what the CNN actually sees using Grad-CAM (gradient heatmaps). Show which part of the leaf the model focuses on. This is real agricultural AI.",
          skills: ["CNN layers", "Data augmentation", "Grad-CAM visualization", "Keras functional API"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "PlantVillage (Kaggle)"
        },
        learn: ["How convolutions detect edges/textures/patterns", "Why data augmentation is free performance", "Model interpretability matters"]
      },
      {
        day: 17,
        title: "Transfer the Knowledge",
        topic: "Transfer Learning",
        emoji: "🔄",
        theory: ["Fine-tuning vs feature extraction", "ImageNet pretrained weights explained", "When to freeze layers, when to unfreeze"],
        project: {
          name: "Indian Cuisine Classifier",
          icon: "🍛",
          desc: "Collect images of 10 Indian dishes (Google/scrape). Fine-tune MobileNetV2 pretrained on ImageNet. Get 95%+ accuracy with 200 images per class — impossible without transfer learning. Build a Gradio demo where you upload a photo and it identifies the dish + suggests recipe pairings.",
          skills: ["MobileNetV2 fine-tuning", "Gradio UI", "Layer freezing", "Custom dataset creation"],
          time: "5 hrs",
          difficulty: "ADVANCED",
          dataset: "Self-collected / Indian Food 101"
        },
        learn: ["Why reusing knowledge beats training from scratch", "The economics of transfer learning", "How to build deployable demos"]
      },
      {
        day: 18,
        title: "Language is Just Sequences",
        topic: "RNNs & LSTMs",
        emoji: "📜",
        theory: ["The vanishing gradient problem", "LSTM gates: forget, input, output", "Sequence generation strategies (greedy, temperature)"],
        project: {
          name: "Hindi News Headline Generator",
          icon: "📰",
          desc: "Train a character-level LSTM on 10,000 real Hindi news headlines (romanized). Generate new headlines at different 'temperatures' (creativity settings). Low temp = boring but plausible. High temp = absolutely unhinged. Build a side-by-side generator showing how temperature controls creativity.",
          skills: ["LSTM", "Character tokenization", "Temperature sampling", "Sequence generation"],
          time: "4–5 hrs",
          difficulty: "ADVANCED",
          dataset: "Scraped/public Hindi news APIs"
        },
        learn: ["How language models generate text", "Why temperature matters for creativity", "The limits of LSTMs (why transformers won)"]
      },
      {
        day: 19,
        title: "Attention is Everything",
        topic: "Transformers & BERT",
        emoji: "🤖",
        theory: ["Self-attention mechanism from scratch", "BERT vs GPT: encoder vs decoder", "Tokenization, positional encoding"],
        project: {
          name: "Hinglish Sentiment Engine",
          icon: "💬",
          desc: "Fine-tune multilingual BERT on Hinglish (Hindi-English code-switched) tweets. Real challenge: code-switching breaks standard NLP pipelines. Build a sentiment API that handles 'yaar ye movie bahut bakwaas thi 💀' correctly. Compare against English-only models to see the gap.",
          skills: ["HuggingFace Transformers", "BERT fine-tuning", "Multilingual NLP", "Code-switched text"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "SemEval Hinglish / Twitter API"
        },
        learn: ["Why attention revolutionized AI", "How BERT understands context", "The real challenge of multilingual AI"]
      },
      {
        day: 20,
        title: "Make Something New",
        topic: "Generative AI",
        emoji: "🎭",
        theory: ["GAN: generator vs discriminator game", "VAE: encoding to latent space", "Diffusion models: the new king"],
        project: {
          name: "Abstract Art Generator",
          icon: "🖼️",
          desc: "Train a DCGAN on WikiArt abstract paintings subset. Watch the GAN learn beauty from static noise over 50 epochs. Log generator/discriminator loss — see the adversarial dance in real-time plots. Compare epoch 1, 10, 25, 50 outputs. The model goes from TV static to something hanging in a gallery.",
          skills: ["DCGAN architecture", "Training loop", "Loss monitoring", "Image generation pipeline"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "WikiArt subset (Kaggle)"
        },
        learn: ["The adversarial training game", "Why GANs are unstable and how to stabilize", "Generative AI foundations"]
      },
      {
        day: 21,
        title: "Week 3 Boss Battle",
        topic: "Vision + Language",
        emoji: "⚔️",
        theory: ["Multimodal AI: vision + language", "CLIP: contrastive learning explained", "Zero-shot vs few-shot classification"],
        project: {
          name: "Visual Meme Search Engine",
          icon: "🔎",
          desc: "Use OpenAI CLIP (via HuggingFace). Build a system where you type a concept — 'crying but make it funny' or 'cats judging humans' — and it searches your meme folder and returns the top 5 matches. Zero labels needed. Pure vision-language embedding magic. Demo this to anyone and watch their jaw drop.",
          skills: ["CLIP model", "Cosine similarity search", "Embedding spaces", "Zero-shot classification"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "Your own image folder"
        },
        learn: ["How multimodal AI works", "The power of shared embedding spaces", "Zero-shot learning in practice"]
      }
    ]
  },
  {
    id: 4,
    title: "Applied AI & Deployment",
    subtitle: "WEEK 04",
    accent: "#FBBF24",
    bg: "rgba(251,191,36,0.06)",
    days: [
      {
        day: 22,
        title: "NLP Goes Pro",
        topic: "Advanced NLP Pipelines",
        emoji: "📋",
        theory: ["Named Entity Recognition (NER)", "Extractive vs abstractive summarization", "Question-answering with context"],
        project: {
          name: "Legal Document Shield",
          icon: "⚖️",
          desc: "Feed any rental agreement or terms-of-service PDF. Use HuggingFace NER to extract parties, dates, obligations. Use summarization to distill clauses. Flag clauses with risky words ('waive all rights', 'perpetual license', 'no refunds'). Output: 'Here are 3 clauses you should NOT sign without a lawyer.' Actually useful.",
          skills: ["HuggingFace pipelines", "PDF parsing (PyMuPDF)", "NER + summarization", "Rule-based flagging"],
          time: "4–5 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "Any real PDF terms of service"
        },
        learn: ["How NLP pipelines compose", "The difference between NER and classification", "Building actually useful AI tools"]
      },
      {
        day: 23,
        title: "Recommend Everything",
        topic: "Recommendation Systems",
        emoji: "🎯",
        theory: ["Collaborative filtering: user-based & item-based", "Content-based filtering", "Matrix factorization with SVD"],
        project: {
          name: "Bollywood Movie Matchmaker",
          icon: "🎬",
          desc: "MovieLens + Bollywood data merged. Build a HYBRID recommender: collaborative (what similar users liked) + content-based (genre/director/actor similarity). Interactive CLI: 'I loved Dil Chahta Hai and 3 Idiots, hate horror' → 10 ranked recommendations with explanations of WHY each was picked.",
          skills: ["Matrix factorization", "Cosine similarity", "Hybrid systems", "Explanation generation"],
          time: "5 hrs",
          difficulty: "INTERMEDIATE",
          dataset: "MovieLens + IMDB Bollywood scrape"
        },
        learn: ["The cold start problem", "Why Netflix recommendations are hard", "How to build explainable recommendations"]
      },
      {
        day: 24,
        title: "AI Sees Your Body",
        topic: "Computer Vision Applications",
        emoji: "🏋️",
        theory: ["Pose estimation: keypoint detection", "Object detection YOLO basics", "Real-time video processing pipeline"],
        project: {
          name: "AI Fitness Form Judge",
          icon: "💪",
          desc: "MediaPipe Pose on your webcam or any exercise video. Detect 33 body keypoints. Calculate joint angles for squats (knee angle, hip angle, back tilt). Give real-time coaching: 'Your left knee is caving inward — track it over your pinky toe.' 'Back tilt is 15° past safe range.' Build a rep counter too.",
          skills: ["MediaPipe", "Joint angle calculation", "Real-time video", "OpenCV", "Rule-based coaching"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "Your webcam / YouTube exercise videos"
        },
        learn: ["How pose estimation works", "The gap between detection and understanding", "Rule-based AI for safety-critical applications"]
      },
      {
        day: 25,
        title: "Talk to Your Data",
        topic: "LLMs & RAG",
        emoji: "🗣️",
        theory: ["Retrieval-Augmented Generation (RAG)", "Vector databases and semantic search", "Prompt engineering patterns"],
        project: {
          name: "Your Personal Book Oracle",
          icon: "📚",
          desc: "Load any book PDF (try something dense like 'Thinking Fast and Slow' or 'Sapiens'). Chunk it, embed with sentence-transformers, store in FAISS vector DB. Build a chat interface: 'What did Kahneman say about loss aversion?' or 'How would this book apply to my startup?' Answers grounded in actual pages — with citations.",
          skills: ["LangChain", "FAISS vector DB", "Sentence transformers", "RAG pipeline", "Context windows"],
          time: "5 hrs",
          difficulty: "ADVANCED",
          dataset: "Any book PDF you own"
        },
        learn: ["Why LLMs hallucinate and how RAG fixes it", "The power of semantic search", "How ChatGPT plugins actually work"]
      },
      {
        day: 26,
        title: "Ship Your Model",
        topic: "ML Deployment & APIs",
        emoji: "🚢",
        theory: ["FastAPI for ML model serving", "Model serialization (pickle, ONNX)", "Docker containerization basics"],
        project: {
          name: "Deployed AI Skin Analysis API",
          icon: "🌐",
          desc: "Take any trained CNN. Wrap it in FastAPI with a /predict endpoint that accepts image upload and returns predictions + confidence scores. Dockerize it (write Dockerfile). Deploy FREE to Render.com. Test with Postman. You now have a REAL, live, public API that anyone can hit. This is how AI products work.",
          skills: ["FastAPI", "Docker", "Model serving", "REST API design", "Cloud deployment"],
          time: "5–6 hrs",
          difficulty: "ADVANCED",
          dataset: "Your Day 16 plant disease model"
        },
        learn: ["The gap between Jupyter notebook and production", "Docker as the universal shipping container", "API design for ML models"]
      },
      {
        day: 27,
        title: "Track Every Experiment",
        topic: "MLOps & Experiment Tracking",
        emoji: "🧪",
        theory: ["MLflow: tracking, models, registry", "Data versioning with DVC", "Why reproducibility is non-negotiable"],
        project: {
          name: "Model Improvement Dashboard",
          icon: "📈",
          desc: "Take any previous project. Retroactively instrument it with MLflow: log ALL experiments (hyperparams, metrics, artifacts, model versions). Run 15+ experiments systematically varying one thing at a time. Build a comparison dashboard showing how accuracy improved. Write a 'Model Changelog' like a software changelog. This is how ML teams actually work.",
          skills: ["MLflow tracking", "Experiment design", "Model registry", "Systematic tuning"],
          time: "4–5 hrs",
          difficulty: "ADVANCED",
          dataset: "Reuse any prior project"
        },
        learn: ["Why experiments without tracking are wasted", "How ML teams collaborate", "The scientific method applied to ML"]
      },
      {
        day: 28,
        title: "Audit Your AI",
        topic: "Responsible AI & Ethics",
        emoji: "⚖️",
        theory: ["Algorithmic bias: sources and types", "Fairness metrics (demographic parity, equalized odds)", "Explainability with SHAP and LIME"],
        project: {
          name: "Loan Approval Bias Autopsy",
          icon: "🔬",
          desc: "UCI Credit/German Credit dataset. Train a loan classifier with high accuracy. Then audit it: use SHAP to explain individual predictions. Does the model discriminate by gender, age, nationality? Calculate fairness metrics across groups. Write an honest 'Model Card' documenting biases found. The model probably fails. That's the lesson.",
          skills: ["SHAP", "LIME", "Fairness metrics", "Model cards", "Bias analysis"],
          time: "4–5 hrs",
          difficulty: "ADVANCED",
          dataset: "UCI German Credit Dataset"
        },
        learn: ["Why 'high accuracy' doesn't mean 'fair'", "How bias enters data and models", "Your ethical responsibility as an ML engineer"]
      },
      {
        day: 29,
        title: "Capstone Build Day",
        topic: "Full-Stack AI Application",
        emoji: "🏗️",
        theory: ["AI system design principles", "UX patterns for AI products", "How to scope a 1-day project to success"],
        project: {
          name: "YOUR Signature AI App",
          icon: "🌟",
          desc: "Pick ONE problem you genuinely care about. Build a polished, end-to-end application: data → model → API → Streamlit/Gradio UI → deployed online for free. It should be something you're proud to explain at a job interview or show your family. No hand-holding. You know enough now.",
          skills: ["Everything from Days 1–28", "Product thinking", "Integration", "Deployment", "Documentation"],
          time: "8–10 hrs",
          difficulty: "CAPSTONE",
          dataset: "Your choice"
        },
        learn: ["How to scope and complete a real project", "The difference between an ML engineer and a Jupyter dabbler", "Your own problem-solving style"]
      },
      {
        day: 30,
        title: "🎓 Demo Day & Launch",
        topic: "Portfolio, Positioning & What's Next",
        emoji: "🚀",
        theory: ["Building an AI portfolio that gets noticed", "Writing project READMEs with impact", "What to study next: RL, MLOps, LLMOps, Research"],
        project: {
          name: "Launch Your AI Identity",
          icon: "🌐",
          desc: "Polish your top 5 projects. Write compelling GitHub READMEs with GIFs, screenshots, and architecture diagrams. Create a 2-minute demo video for your capstone. Write a LinkedIn post about your 30-day journey (these get thousands of views). Apply to 3 AI/ML roles or internships. You graduated.",
          skills: ["Git + GitHub", "Technical writing", "Screen recording", "Personal branding", "Job applications"],
          time: "5–6 hrs",
          difficulty: "CAPSTONE",
          dataset: "Everything you built"
        },
        learn: ["How to present technical work to non-technical audiences", "What hiring managers in AI actually look for", "Your next 90-day learning path"]
      }
    ]
  }
];

const TOOLS = [
  { name: "Python 3.11+", role: "Core Language", icon: "🐍" },
  { name: "VS Code + Jupyter", role: "IDE", icon: "💻" },
  { name: "NumPy + Pandas", role: "Data Layer", icon: "📊" },
  { name: "Scikit-learn", role: "Classical ML", icon: "⚙️" },
  { name: "TensorFlow/Keras", role: "Deep Learning", icon: "🧠" },
  { name: "HuggingFace 🤗", role: "Transformers + LLMs", icon: "🤖" },
  { name: "FastAPI", role: "Model Serving", icon: "🚀" },
  { name: "Gradio/Streamlit", role: "Demo UIs", icon: "🖼️" },
  { name: "MLflow", role: "Experiment Tracking", icon: "📈" },
  { name: "Docker", role: "Containerization", icon: "📦" },
  { name: "FAISS", role: "Vector Search", icon: "🔍" },
  { name: "Git + GitHub", role: "Version Control", icon: "🌿" },
];

const DAILY_ROUTINE = [
  { time: "6:30 AM", task: "Read theory / watch one video lecture", duration: "30 min", icon: "☀️" },
  { time: "9:00 AM", task: "Code the day's mini-project (core build)", duration: "2–3 hrs", icon: "💻" },
  { time: "12:30 PM", task: "Debug, refine, add one creative twist", duration: "1 hr", icon: "🔧" },
  { time: "7:00 PM", task: "Write README, push to GitHub", duration: "30 min", icon: "📝" },
  { time: "9:30 PM", task: "Note 3 things learned + 1 question for tomorrow", duration: "15 min", icon: "🌙" },
];

const DIFF_COLORS = {
  "BEGINNER": "#00FFB2",
  "INTERMEDIATE": "#FBBF24",
  "ADVANCED": "#FF6B6B",
  "CAPSTONE": "#A78BFA"
};

export default function AIMLRoadmap() {
  const [activeWeek, setActiveWeek] = useState(0);
  const [activeDay, setActiveDay] = useState(null);
  const [tab, setTab] = useState("roadmap");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setTimeout(() => setMounted(true), 100);
  }, []);

  const currentWeek = WEEKS[activeWeek];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#07070E",
      color: "#D4D4D4",
      fontFamily: "'Courier New', 'Lucida Console', monospace",
      overflowX: "hidden",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800;900&display=swap');
        ::-webkit-scrollbar { width: 4px; background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
        .day-card:hover { transform: translateY(-2px); border-color: inherit !important; }
        .day-card { transition: all 0.2s ease; cursor: pointer; }
        .tab-btn:hover { opacity: 1 !important; }
        .tool-chip:hover { transform: scale(1.03); }
        .week-btn:hover { opacity: 1 !important; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        @keyframes scanline { 0% { top: -10%; } 100% { top: 110%; } }
        .animate-in { animation: fadeUp 0.4s ease forwards; }
        .scanline { position: fixed; top: -10%; left: 0; right: 0; height: 2px; background: linear-gradient(transparent, rgba(0,255,178,0.04), transparent); pointer-events: none; z-index: 9999; animation: scanline 8s linear infinite; }
      `}</style>

      {/* Scanline effect */}
      <div className="scanline" />

      {/* Grid background */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0,
        backgroundImage: "linear-gradient(rgba(0,255,178,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,178,0.015) 1px, transparent 1px)",
        backgroundSize: "48px 48px"
      }} />

      <div style={{ position: "relative", zIndex: 1 }}>

        {/* HERO HEADER */}
        <header style={{
          padding: "48px 32px 40px",
          borderBottom: "1px solid #1a1a2e",
          background: "linear-gradient(180deg, #0D0D1A 0%, #07070E 100%)",
          position: "relative",
          overflow: "hidden"
        }}>
          {/* Corner decorations */}
          <div style={{ position: "absolute", top: 16, left: 16, width: 40, height: 40, borderTop: "2px solid #00FFB2", borderLeft: "2px solid #00FFB2" }} />
          <div style={{ position: "absolute", top: 16, right: 16, width: 40, height: 40, borderTop: "2px solid #00FFB2", borderRight: "2px solid #00FFB2" }} />
          <div style={{ position: "absolute", bottom: 16, left: 16, width: 40, height: 40, borderBottom: "2px solid #00FFB2", borderLeft: "2px solid #00FFB2" }} />
          <div style={{ position: "absolute", bottom: 16, right: 16, width: 40, height: 40, borderBottom: "2px solid #00FFB2", borderRight: "2px solid #00FFB2" }} />

          <div style={{ maxWidth: 1000, margin: "0 auto", textAlign: "center" }}>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              background: "rgba(0,255,178,0.08)", border: "1px solid rgba(0,255,178,0.3)",
              padding: "4px 16px", marginBottom: 24,
              fontSize: 10, letterSpacing: 4, color: "#00FFB2", textTransform: "uppercase"
            }}>
              <span style={{ animation: "pulse 2s ease infinite", display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "#00FFB2" }} />
              SYSTEM ONLINE — 30-DAY PROGRAM INITIATED
            </div>

            <h1 style={{
              fontFamily: "'Syne', sans-serif",
              fontSize: "clamp(36px, 7vw, 72px)",
              fontWeight: 900,
              lineHeight: 1.0,
              margin: "0 0 16px",
              letterSpacing: -2
            }}>
              <span style={{ color: "#00FFB2" }}>AI / ML</span>
              <br />
              <span style={{ color: "#fff" }}>MASTERY</span>
              <br />
              <span style={{ color: "#FF6B6B" }}>BOOTCAMP</span>
            </h1>

            <p style={{
              color: "#666", fontSize: 13, lineHeight: 2,
              maxWidth: 560, margin: "0 auto 36px",
              fontFamily: "'Space Mono', monospace"
            }}>
              30 days. 30 real projects. Zero boring MNIST-only repetition.<br />
              Build things you'd actually show someone.
            </p>

            {/* Stats */}
            <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap", marginBottom: 40 }}>
              {[
                { v: "30", l: "Days", c: "#00FFB2" },
                { v: "30", l: "Projects", c: "#FF6B6B" },
                { v: "4", l: "Phases", c: "#A78BFA" },
                { v: "12+", l: "Libraries", c: "#FBBF24" },
                { v: "1", l: "Portfolio", c: "#00FFB2" },
              ].map(s => (
                <div key={s.l} style={{
                  background: "rgba(255,255,255,0.03)",
                  border: `1px solid ${s.c}30`,
                  padding: "14px 20px", textAlign: "center", minWidth: 80,
                  transition: "all 0.2s"
                }}>
                  <div style={{ fontSize: 28, fontWeight: 900, color: s.c, fontFamily: "'Syne', sans-serif" }}>{s.v}</div>
                  <div style={{ fontSize: 9, color: "#555", letterSpacing: 3, textTransform: "uppercase", marginTop: 2 }}>{s.l}</div>
                </div>
              ))}
            </div>

            {/* Main nav */}
            <div style={{ display: "flex", gap: 4, justifyContent: "center", flexWrap: "wrap" }}>
              {["roadmap", "tools", "routine"].map(t => (
                <button key={t} className="tab-btn" onClick={() => setTab(t)} style={{
                  padding: "8px 20px",
                  background: tab === t ? "#00FFB2" : "transparent",
                  color: tab === t ? "#07070E" : "#666",
                  border: `1px solid ${tab === t ? "#00FFB2" : "#222"}`,
                  cursor: "pointer", fontSize: 10, letterSpacing: 2,
                  textTransform: "uppercase", fontFamily: "'Space Mono', monospace",
                  fontWeight: tab === t ? 700 : 400, transition: "all 0.2s",
                  opacity: tab === t ? 1 : 0.7
                }}>
                  {t === "roadmap" ? "📅 30-Day Roadmap" : t === "tools" ? "🛠️ Tool Stack" : "⏰ Daily Routine"}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* MAIN CONTENT */}
        <main style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px" }}>

          {/* ── ROADMAP TAB ── */}
          {tab === "roadmap" && (
            <div>
              {/* Week selector */}
              <div style={{ display: "flex", gap: 8, marginBottom: 32, flexWrap: "wrap" }}>
                {WEEKS.map((w, i) => (
                  <button key={i} className="week-btn" onClick={() => { setActiveWeek(i); setActiveDay(null); }} style={{
                    padding: "10px 18px", flex: 1, minWidth: 180,
                    background: activeWeek === i ? w.bg : "transparent",
                    border: `1px solid ${activeWeek === i ? w.accent : "#222"}`,
                    color: activeWeek === i ? w.accent : "#555",
                    cursor: "pointer", textAlign: "left",
                    fontFamily: "'Space Mono', monospace",
                    fontSize: 10, letterSpacing: 1, transition: "all 0.2s",
                    opacity: activeWeek === i ? 1 : 0.7
                  }}>
                    <div style={{ fontSize: 8, letterSpacing: 3, marginBottom: 4, opacity: 0.7 }}>{w.subtitle}</div>
                    <div style={{ fontWeight: 700, fontSize: 11 }}>{w.title}</div>
                  </button>
                ))}
              </div>

              {/* Day detail panel */}
              {activeDay !== null && (
                <div className="animate-in" style={{
                  background: "rgba(255,255,255,0.02)",
                  border: `1px solid ${currentWeek.accent}40`,
                  marginBottom: 32,
                  padding: "28px 32px",
                  position: "relative",
                  overflow: "hidden"
                }}>
                  {/* Accent bar */}
                  <div style={{
                    position: "absolute", left: 0, top: 0, bottom: 0, width: 3,
                    background: currentWeek.accent
                  }} />

                  <button onClick={() => setActiveDay(null)} style={{
                    position: "absolute", top: 16, right: 16,
                    background: "transparent", border: "1px solid #333",
                    color: "#666", cursor: "pointer", padding: "4px 10px",
                    fontSize: 10, fontFamily: "'Space Mono', monospace"
                  }}>✕ CLOSE</button>

                  {(() => {
                    const day = currentWeek.days[activeDay];
                    return (
                      <div>
                        <div style={{ marginBottom: 24 }}>
                          <div style={{
                            display: "flex", alignItems: "center", gap: 12, marginBottom: 8, flexWrap: "wrap"
                          }}>
                            <span style={{ fontSize: 28 }}>{day.emoji}</span>
                            <div>
                              <div style={{
                                fontSize: 9, color: currentWeek.accent, letterSpacing: 3,
                                textTransform: "uppercase", marginBottom: 4
                              }}>DAY {day.day} — {day.topic}</div>
                              <h2 style={{
                                fontFamily: "'Syne', sans-serif",
                                fontSize: 24, fontWeight: 900, color: "#fff", margin: 0
                              }}>{day.title}</h2>
                            </div>
                          </div>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
                          {/* Theory */}
                          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid #1a1a2e", padding: "18px 20px" }}>
                            <div style={{ fontSize: 9, color: "#666", letterSpacing: 3, textTransform: "uppercase", marginBottom: 12 }}>📖 Theory to Study</div>
                            {day.theory.map((t, i) => (
                              <div key={i} style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "flex-start" }}>
                                <span style={{ color: currentWeek.accent, flexShrink: 0, fontSize: 10 }}>→</span>
                                <span style={{ fontSize: 11, lineHeight: 1.6, color: "#aaa" }}>{t}</span>
                              </div>
                            ))}
                          </div>

                          {/* What you'll learn */}
                          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid #1a1a2e", padding: "18px 20px" }}>
                            <div style={{ fontSize: 9, color: "#666", letterSpacing: 3, textTransform: "uppercase", marginBottom: 12 }}>🧠 You'll Understand</div>
                            {day.learn.map((l, i) => (
                              <div key={i} style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "flex-start" }}>
                                <span style={{ color: "#FBBF24", flexShrink: 0, fontSize: 10 }}>✓</span>
                                <span style={{ fontSize: 11, lineHeight: 1.6, color: "#aaa" }}>{l}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Project card */}
                        <div style={{
                          background: currentWeek.bg,
                          border: `1px solid ${currentWeek.accent}50`,
                          padding: "24px 28px"
                        }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
                            <div>
                              <div style={{ fontSize: 9, color: currentWeek.accent, letterSpacing: 3, textTransform: "uppercase", marginBottom: 6 }}>🔨 TODAY'S PROJECT</div>
                              <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 900, color: "#fff" }}>
                                {day.project.icon} {day.project.name}
                              </div>
                            </div>
                            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                              <span style={{
                                background: `${DIFF_COLORS[day.project.difficulty]}20`,
                                border: `1px solid ${DIFF_COLORS[day.project.difficulty]}50`,
                                color: DIFF_COLORS[day.project.difficulty],
                                padding: "3px 10px", fontSize: 8, letterSpacing: 2
                              }}>{day.project.difficulty}</span>
                              <span style={{
                                background: "rgba(255,255,255,0.04)", border: "1px solid #333",
                                color: "#888", padding: "3px 10px", fontSize: 8, letterSpacing: 1
                              }}>⏱ {day.project.time}</span>
                            </div>
                          </div>

                          <p style={{ color: "#ccc", fontSize: 12, lineHeight: 1.9, marginBottom: 20 }}>
                            {day.project.desc}
                          </p>

                          <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                            <div style={{ flex: 1, minWidth: 200 }}>
                              <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>Skills Used</div>
                              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                                {day.project.skills.map(s => (
                                  <span key={s} style={{
                                    background: "rgba(255,255,255,0.04)", border: "1px solid #2a2a3e",
                                    color: "#888", padding: "3px 8px", fontSize: 9, letterSpacing: 1
                                  }}>{s}</span>
                                ))}
                              </div>
                            </div>
                            <div>
                              <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>Dataset</div>
                              <span style={{ color: currentWeek.accent, fontSize: 11 }}>📂 {day.project.dataset}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* Day grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12 }}>
                {currentWeek.days.map((day, i) => (
                  <div key={day.day} className="day-card"
                    onClick={() => setActiveDay(activeDay === i ? null : i)}
                    style={{
                      background: activeDay === i ? currentWeek.bg : "rgba(255,255,255,0.02)",
                      border: `1px solid ${activeDay === i ? currentWeek.accent : "#1a1a2e"}`,
                      padding: "20px 22px",
                      animation: `fadeUp 0.3s ease ${i * 0.05}s both`,
                      position: "relative", overflow: "hidden"
                    }}>

                    {/* Day number */}
                    <div style={{
                      position: "absolute", top: 12, right: 16,
                      fontFamily: "'Syne', sans-serif",
                      fontSize: 36, fontWeight: 900,
                      color: activeDay === i ? currentWeek.accent : "#111",
                      lineHeight: 1, transition: "color 0.2s", userSelect: "none"
                    }}>{String(day.day).padStart(2, "0")}</div>

                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                      <span style={{ fontSize: 20 }}>{day.emoji}</span>
                      <span style={{
                        fontSize: 8, color: activeDay === i ? currentWeek.accent : "#444",
                        letterSpacing: 2, textTransform: "uppercase"
                      }}>DAY {day.day}</span>
                    </div>

                    <h3 style={{
                      fontFamily: "'Syne', sans-serif",
                      fontSize: 15, fontWeight: 800, color: "#fff",
                      margin: "0 0 4px", paddingRight: 36, lineHeight: 1.3
                    }}>{day.title}</h3>

                    <div style={{ fontSize: 10, color: "#555", marginBottom: 12 }}>{day.topic}</div>

                    <div style={{
                      display: "flex", gap: 6, alignItems: "center",
                      padding: "8px 10px",
                      background: `${currentWeek.accent}08`,
                      border: `1px solid ${currentWeek.accent}15`
                    }}>
                      <span style={{ fontSize: 14 }}>{day.project.icon}</span>
                      <div>
                        <div style={{ fontSize: 10, fontWeight: 700, color: "#ccc" }}>{day.project.name}</div>
                        <div style={{ fontSize: 8, color: "#555", marginTop: 1 }}>{day.project.time}</div>
                      </div>
                    </div>

                    <div style={{ marginTop: 10, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{
                        fontSize: 8, letterSpacing: 1,
                        color: DIFF_COLORS[day.project.difficulty],
                        border: `1px solid ${DIFF_COLORS[day.project.difficulty]}40`,
                        padding: "2px 6px"
                      }}>{day.project.difficulty}</span>
                      <span style={{ fontSize: 9, color: "#444" }}>
                        {activeDay === i ? "CLICK TO CLOSE ▲" : "CLICK TO EXPAND ▼"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Progress bar */}
              <div style={{ marginTop: 32, padding: "20px 24px", background: "rgba(255,255,255,0.02)", border: "1px solid #1a1a2e" }}>
                <div style={{ fontSize: 9, color: "#444", letterSpacing: 3, textTransform: "uppercase", marginBottom: 12 }}>COURSE PROGRESS</div>
                <div style={{ display: "flex", gap: 3 }}>
                  {WEEKS.map((w, wi) =>
                    w.days.map((d, di) => (
                      <div key={`${wi}-${di}`} style={{
                        flex: 1, height: 6,
                        background: wi < activeWeek ? w.accent : (wi === activeWeek ? `${w.accent}60` : "#1a1a2e"),
                        transition: "background 0.3s"
                      }} />
                    ))
                  )}
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
                  {WEEKS.map((w, i) => (
                    <div key={i} style={{ fontSize: 8, color: i <= activeWeek ? w.accent : "#333", letterSpacing: 1 }}>{w.subtitle}</div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── TOOLS TAB ── */}
          {tab === "tools" && (
            <div className="animate-in">
              <div style={{ marginBottom: 32 }}>
                <div style={{ fontSize: 9, color: "#00FFB2", letterSpacing: 4, textTransform: "uppercase", marginBottom: 8 }}>SETUP FIRST — BEFORE DAY 1</div>
                <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 28, fontWeight: 900, color: "#fff", margin: "0 0 16px" }}>
                  Your AI/ML Stack
                </h2>
                <p style={{ color: "#666", fontSize: 12, lineHeight: 1.8, maxWidth: 560 }}>
                  Install these tools on Day 0. Every project in the 30 days uses at least one of these. Don't skip setup — debugging your environment mid-project kills momentum.
                </p>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12, marginBottom: 40 }}>
                {TOOLS.map((tool, i) => (
                  <div key={tool.name} className="tool-chip" style={{
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid #1a1a2e",
                    padding: "18px 20px",
                    transition: "all 0.2s",
                    animation: `fadeUp 0.3s ease ${i * 0.04}s both`
                  }}>
                    <div style={{ fontSize: 24, marginBottom: 10 }}>{tool.icon}</div>
                    <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 14, fontWeight: 800, color: "#fff", marginBottom: 4 }}>{tool.name}</div>
                    <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, textTransform: "uppercase" }}>{tool.role}</div>
                  </div>
                ))}
              </div>

              {/* Install command */}
              <div style={{ background: "#0D0D1A", border: "1px solid #1a1a2e", padding: "24px 28px" }}>
                <div style={{ fontSize: 9, color: "#00FFB2", letterSpacing: 3, textTransform: "uppercase", marginBottom: 16 }}>TERMINAL — RUN THIS FIRST</div>
                <pre style={{
                  color: "#00FFB2", fontSize: 11, lineHeight: 2,
                  margin: 0, whiteSpace: "pre-wrap", fontFamily: "'Space Mono', monospace"
                }}>{`pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm scipy
pip install tensorflow keras torch torchvision
pip install transformers datasets huggingface-hub
pip install fastapi uvicorn gradio streamlit
pip install mlflow langchain faiss-cpu
pip install opencv-python mediapipe
pip install shap lime`}
                </pre>
              </div>

              {/* Learning path */}
              <div style={{ marginTop: 32 }}>
                <div style={{ fontSize: 9, color: "#FF6B6B", letterSpacing: 4, textTransform: "uppercase", marginBottom: 20 }}>LEARNING PATH — HOW IT BUILDS</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
                  {[
                    { phase: "Week 1", title: "Foundation", desc: "Python → NumPy → Pandas → Viz → Statistics → Math. You need these as reflexes, not knowledge.", color: "#00FFB2" },
                    { phase: "Week 2", title: "Classical ML", desc: "Linear models → Trees → Ensembles → Clustering → Pipelines. The algorithms that still power 80% of production ML.", color: "#FF6B6B" },
                    { phase: "Week 3", title: "Deep Learning", desc: "Neural nets → CNNs → Transfer learning → RNNs → Transformers → Generative AI. The sexy stuff, now you're ready for it.", color: "#A78BFA" },
                    { phase: "Week 4", title: "Ship It", desc: "NLP → Recommenders → CV apps → LLMs + RAG → APIs → MLOps → Ethics → Deploy. Build things the world can actually use.", color: "#FBBF24" },
                  ].map((p, i) => (
                    <div key={i} style={{ display: "flex", gap: 0 }}>
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginRight: 20 }}>
                        <div style={{ width: 12, height: 12, borderRadius: "50%", background: p.color, flexShrink: 0, marginTop: 18 }} />
                        {i < 3 && <div style={{ width: 1, flex: 1, background: "#222", marginTop: 4 }} />}
                      </div>
                      <div style={{
                        flex: 1, padding: "16px 20px",
                        background: "rgba(255,255,255,0.02)", border: "1px solid #1a1a2e",
                        marginBottom: 8
                      }}>
                        <div style={{ fontSize: 9, color: p.color, letterSpacing: 3, textTransform: "uppercase", marginBottom: 4 }}>{p.phase} — {p.title}</div>
                        <div style={{ fontSize: 12, color: "#888", lineHeight: 1.7 }}>{p.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── ROUTINE TAB ── */}
          {tab === "routine" && (
            <div className="animate-in">
              <div style={{ marginBottom: 32 }}>
                <div style={{ fontSize: 9, color: "#00FFB2", letterSpacing: 4, textTransform: "uppercase", marginBottom: 8 }}>REPEAT THIS EVERY SINGLE DAY</div>
                <h2 style={{ fontFamily: "'Syne', sans-serif", fontSize: 28, fontWeight: 900, color: "#fff", margin: "0 0 16px" }}>
                  The Daily System
                </h2>
                <p style={{ color: "#666", fontSize: 12, lineHeight: 1.8, maxWidth: 600 }}>
                  30 days only works if you're consistent. This routine is designed for someone with a 4–5 hour window. Adjust times, keep the structure.
                </p>
              </div>

              {/* Routine blocks */}
              <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 48 }}>
                {DAILY_ROUTINE.map((r, i) => (
                  <div key={i} className="animate-in" style={{
                    display: "flex", gap: 20, alignItems: "center",
                    background: "rgba(255,255,255,0.02)", border: "1px solid #1a1a2e",
                    padding: "18px 24px",
                    animation: `fadeUp 0.3s ease ${i * 0.07}s both`,
                    borderLeft: `3px solid #00FFB2`
                  }}>
                    <div style={{ fontSize: 24, flexShrink: 0 }}>{r.icon}</div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, textTransform: "uppercase", marginBottom: 4 }}>{r.time}</div>
                      <div style={{ fontSize: 13, color: "#ccc", fontWeight: 600 }}>{r.task}</div>
                    </div>
                    <div style={{
                      background: "rgba(0,255,178,0.08)", border: "1px solid rgba(0,255,178,0.2)",
                      color: "#00FFB2", padding: "4px 12px", fontSize: 9,
                      letterSpacing: 1, textTransform: "uppercase", flexShrink: 0
                    }}>{r.duration}</div>
                  </div>
                ))}
              </div>

              {/* Rules */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 32 }}>
                <div style={{ background: "rgba(0,255,178,0.04)", border: "1px solid rgba(0,255,178,0.2)", padding: "24px" }}>
                  <div style={{ fontSize: 9, color: "#00FFB2", letterSpacing: 3, textTransform: "uppercase", marginBottom: 16 }}>✅ DO THESE</div>
                  {[
                    "Push every project to GitHub, even messy code",
                    "Run the code before you understand it fully",
                    "Break the project, fix it, understand why",
                    "Add one creative twist to every project",
                    "Write one README per week minimum",
                    "Ask 'What would break this model?' after every build",
                  ].map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, marginBottom: 10, alignItems: "flex-start" }}>
                      <span style={{ color: "#00FFB2", fontSize: 10, flexShrink: 0 }}>→</span>
                      <span style={{ fontSize: 11, color: "#aaa", lineHeight: 1.6 }}>{r}</span>
                    </div>
                  ))}
                </div>
                <div style={{ background: "rgba(255,107,107,0.04)", border: "1px solid rgba(255,107,107,0.2)", padding: "24px" }}>
                  <div style={{ fontSize: 9, color: "#FF6B6B", letterSpacing: 3, textTransform: "uppercase", marginBottom: 16 }}>🚫 DON'T DO THESE</div>
                  {[
                    "Don't skip days — do 30 min minimum if busy",
                    "Don't copy-paste code without understanding it",
                    "Don't move on if you can't explain the concept",
                    "Don't study only theory with no coding",
                    "Don't wait until code is 'perfect' to push",
                    "Don't use ChatGPT to solve problems before trying",
                  ].map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, marginBottom: 10, alignItems: "flex-start" }}>
                      <span style={{ color: "#FF6B6B", fontSize: 10, flexShrink: 0 }}>✕</span>
                      <span style={{ fontSize: 11, color: "#aaa", lineHeight: 1.6 }}>{r}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* What to do if stuck */}
              <div style={{ background: "rgba(167,139,250,0.04)", border: "1px solid rgba(167,139,250,0.2)", padding: "24px 28px" }}>
                <div style={{ fontSize: 9, color: "#A78BFA", letterSpacing: 3, textTransform: "uppercase", marginBottom: 16 }}>WHEN YOU'RE STUCK (YOU WILL BE)</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12 }}>
                  {[
                    { step: "1", text: "Read the error message 3x slowly" },
                    { step: "2", text: "Print intermediate variables" },
                    { step: "3", text: "Rubber duck debug (explain aloud)" },
                    { step: "4", text: "Search exact error on Stack Overflow" },
                    { step: "5", text: "Check docs for the function you're using" },
                    { step: "6", text: "Then ask Claude/ChatGPT with full context" },
                  ].map(s => (
                    <div key={s.step} style={{
                      display: "flex", gap: 10, alignItems: "flex-start",
                      background: "rgba(255,255,255,0.02)", padding: "12px 14px",
                      border: "1px solid #1a1a2e"
                    }}>
                      <span style={{
                        fontFamily: "'Syne', sans-serif", fontSize: 18, fontWeight: 900,
                        color: "#A78BFA", flexShrink: 0, lineHeight: 1
                      }}>{s.step}</span>
                      <span style={{ fontSize: 11, color: "#888", lineHeight: 1.6 }}>{s.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

        </main>

        {/* Footer */}
        <footer style={{
          borderTop: "1px solid #1a1a2e",
          padding: "24px 32px",
          textAlign: "center",
          background: "rgba(0,0,0,0.3)"
        }}>
          <div style={{ fontSize: 9, color: "#333", letterSpacing: 3, textTransform: "uppercase" }}>
            AI/ML MASTERY BOOTCAMP — 30 DAYS — BUILD REAL THINGS
          </div>
        </footer>
      </div>
    </div>
  );
}
