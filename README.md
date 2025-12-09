# Sentiment Analysis API
<div align="center">
    <img src="https://litslink.com/wp-content/uploads/2021/03/sentiment-featured.jpg"/>
<!--     <img src="https://t4.ftcdn.net/jpg/08/92/22/07/360_F_892220774_JALjFtTUwSQBsgd1dOUvagKlhPYrDTKU.jpg"/> -->
</div>

## <img align="center" width="80px" src="https://cdn.dribbble.com/users/35253/screenshots/3984334/ideate_dribb.gif"> Table of Contents
- <a href="#Overview">Overview</a>
- <a href="#tools">Built Using</a>
- <a href="#folder-structure">Folder Structure</a>
- <a href="#documentation">API Documentation</a>
- <a href="#started">Get Started</a>
    - <a href="#native">Native</a>
    - <a href="#docker">Docker</a>
    - <a href="#docker-compose">Docker Compose</a>
- <a href="#tests">Tests</a>
- <a href="#gsoc">GSoC Docs</a>
- <a href="#license">License</a>


<!-- Overview -->
## <img align="center" width="70px" src="https://cdn.dribbble.com/users/1501052/screenshots/4545496/media/13e279b5c3bd2e8f79067239da3d8633.gif"> Overview <a id="Overview"></a>

This project is a standalone backend solution for sentiment analysis of video and audio files. It processes both complete files and segmented audio from one-sided or multi-speaker videos. The system provides sentiment analysis for each segment of the audio or video, including a transcription and its corresponding sentiment for each timestamp.

Developed as one of the main projects for RuxAiLab for GSoC '24, this tool integrates with the RuxAiLab project through [RuxAiLab PRs](#integration_to_ruxailab). The pipeline involves:

1. **Input:** Video or audio files are uploaded to the system.
2. **Pipeline Modules:** 
   - **Audio Extraction:** Extract audio from video files using MoviePy.
   - **Transcription:** Transcribe the audio using Whisper.
   - **Sentiment Analysis:** Analyze the transcription for sentiment using RoBERTa. Sentiment labels are **POS** (Positive), **NEU** (Neutral), and **NEG** (Negative).
3. **Output:** Generate a detailed report that includes the transcription along with sentiment analysis for each timestamp.

![Pipeline](https://github.com/user-attachments/assets/bdd0ca21-9f90-41bd-83ac-bbc6494fd653)

This project aims to enhance RuxAiLab's capabilities by providing detailed sentiment insights along with transcriptions for better understanding and analysis for users of RuxAiLab.

<!-- Tools -->
## <img  align= center width =60px  height =70px src="https://media4.giphy.com/media/ux6vPam8BubuCxbW20/giphy.gif?cid=6c09b952gi267xsujaqufpqwuzeqhbi88q0ohj83jwv6dpls&ep=v1_stickers_related&rid=giphy.gif&ct=s"> Built Using<a id="tools"></a>
<table style="border-collapse: collapse; border: none;">
  <tr>
    <td><img height="60" src="https://cdn-icons-png.flaticon.com/512/6124/6124995.png"/></td>
    <td><img height="60" src="https://miro.medium.com/v2/resize:fit:1400/0*adyeTInZ7lebNANK.png"/></td>
    <td><img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png"/></td>
    <td><img height="70" src="https://cdn.worldvectorlogo.com/logos/docker.svg"/></td>
    <td><img height="60" src="https://miro.medium.com/v2/resize:fit:438/1*dQvABiWzbE28OTPYjzElKw.png"/></td>
    <td><img height="80" src="https://miro.medium.com/v2/resize:fit:800/1*HTYDFA422w071hjbMuMqGA.png"/></td>
  </tr>
</table>


<!-- Folder Structure -->
## <img align="center" width="70px" src="https://media.lordicon.com/icons/wired/lineal/120-folder.gif"> Folder Structure <a id="folder-structure"></a>

Here is the folder structure of the Sentiment Analysis API project:

```
sentiment-analysis-api/
├── app/
│   ├── __init__.py         # Initializes the app and its components
│   ├── config.py           # Configuration for app settings
│   ├── data/               # Data handling module
│   ├── models/             # Contains the models for sentiment analysis
│   ├── routes/             # Defines the routes for the API
│   ├── services/           # Contains the business logic services
│   └── ...                 # Additional app files
├── docs/                   # Documentation files
├── docker-compose.yml      # Defines Docker Compose configuration
├── Dockerfile              # Defines Docker container setup
├── .env                    # Environment variables
├── pytest.ini              # Pytest configuration
├── README.md               # Project documentation
├── CONTRIBUTING.md.md               # Project documentation
├── requirements.txt        # Lists required Python dependencies
├── run.py                  # Entry point to run the app
├── samples/                # Sample input files for testing
├── static/                 # Static files (e.g., images)
├── tests/                  # Contains unit and integration tests
│   ├── coverage/           # Coverage reports
│   ├── integration/        # Integration tests
│   ├── unit/               # Unit tests
│   └── ...                 # Other test files
└── ...                     # Other files
```
This structure helps separate the application logic, configuration files, test files, and Docker-related configurations.

### 🧩 Requirements
- **Python:** 3.12
- **pip:** 24

<!-- Getting Started -->
## <img align="center" width="60px" height="60px" src="https://media3.giphy.com/media/wuZWV7keWqi2jJGzdB/giphy.gif?cid=6c09b952wp4ev7jtywg3j6tt7ec7vr3piiwql2vhrlsgydyz&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=s"> Get Started <a id="started"></a>

<!-- Native -->
### <img align="center" height="40px" width="50px" src="https://media.lordicon.com/icons/wired/outline/743-web-code.gif"> Native <a id="native"></a>
1. **Clone the Repository**
    ```bash
    ~$ git clone https://github.com/ruxailab/sentiment-analysis-api.git
    ~$ cd sentiment-analysis-api
    ```
2. **Create the Virtual Environment**
    - **Linux**
        ```bash
         ~/sentiment-analysis-api$ python3 -m venv env
        ```
    - **Windows**
        ```bash
         python -m venv env
        ```    
4. **Activate the virtual environment**
    - **Linux**
        ```bash
         ~/sentiment-analysis-api$ source env/bin/activate
        ```
    - **Windows**
        ```bash
          env\Scripts\activate
        ```
6. **Install pip Dependencies**
    ```bash
     (env) ~/sentiment-analysis-api$ pip install -r requirements.txt
    ```
7. **Install <a href="https://www.ffmpeg.org/" target="_blank">FFmpeg</a>**
    - **Linux**
        ```bash
        ~$ apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg
        ```
    - **Windows (Win10)**
        - Follow Tutorial [here](htps://www.youtube.com/watch?v=IECI72XEox0)
        - Add FFmpeg to the system path.    
8. **Run Flask App**
   -    In Debug Mode  [port 8001]
        ```bash
         ~/sentiment-analysis-api$ python3 -m run
        ```
9. **Run API Documentation**
     - Access the API documentation at: http://localhost:8001/apidocs
   
<!-- Docker -->
### <img align="center" width="60px" src="https://miro.medium.com/v2/resize:fit:1400/1*wXtyhpOL5NK_w39UvZpADQ.gif"> Docker <a id="docker"></a>
1. **Install Docker**
    ```bash
    ~$ sudo apt  install docker.io  
    ~$ docker --version    ## Docker version 26.1.3, build 26.1.3-0ubuntu1~22.04.1
    ```
2. **Clone the Repository**
    ```bash
    ~$ git clone https://github.com/ruxailab/sentiment-analysis-api.git
    ~$ cd sentiment-analysis-api
    ```
3. **Build Image**
    ```bash
    ~/sentiment-analysis-api$ docker build -t sentiment_analysis_api .
    ```
6. **Start Docker Conatiner (Port 8001)**
    - **New Container**  
    ```bash
    ~/sentiment-analysis-api$ docker run -d --name sentiment_analysis_api_app -p 8001:8001 -v ./:/sentiment-analysis-api sentiment_analysis_api
    ```
    - **Existing Container**
    ```bash
    ~/sentiment-analysis-api$ docker start --name sentiment_analysis_api_app
    ```
7. **Run API Documentation**
     - Access the API documentation at: http://localhost:8001/apidocs


<!-- Docker Compose -->
### <img align="center" width="60px" src="https://github.com/user-attachments/assets/af017ff7-3275-4ae5-b706-d3a3e85bd9bf">Docker Compose<a id="docker-compose"></a>
1. **Install Docker Compose**
    ```bash
    ~$ sudo apt install docker-compose-v2
    ~$ docker compose --version   ## Docker Compose version 2.27.1+ds1-0ubuntu1~22.04.1
    ```
2. **Clone the Repository**     
    ```bash
    ~$ git clone https://github.com/ruxailab/sentiment-analysis-api.git
    ~$ cd sentiment-analysis-api
    ```
3. **Build Image Using Docker Compose**
    ```bash
     ~/sentiment-analysis-api$ docker compose build
    ```
4. **Start Docker Container**
   - **New Container**  
    ```bash
    ~/sentiment-analysis-api$ docker compose up
    ```
   - **Existing Container**
    ```bash
    ~/sentiment-analysis-api$ docker compose start
    ```
6. **Run API Documentation**
     - Access the API documentation at: http://localhost:8001/apidocs
   

<!-- API Documentation -->
## <img align="center" width="80px" src="https://cdn.dribbble.com/users/1874602/screenshots/5647628/send-icon.gif"> API Documentation <a id="documentaion"></a>
You can access the API documentation at [http://localhost:8001/apidocs](http://localhost:8001/apidocs) after running the Flask App.

<!-- Postman -->
### Postman Collection
For testing the API endpoints, you can use the following Postman collection:
- [RuxAiLab Sentiment Analysis APIs Postman Collection](https://www.postman.com/interstellar-shadow-582340/workspace/ruxaailab/collection/31975349-d17198fa-8c4f-41d4-9870-1dc6e7443bc3) 
 


<!-- Tests -->
## <img align="center" width="60px" src="https://cdn-icons-png.flaticon.com/512/10435/10435234.png" > Tests <a id="tests"></a>

1. **Unit Tests**
    - Run Data Layer unit tests using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage run  -m pytest ./tests/unit/test_data/                            
        ```
    - Run Service Layer unit tests using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage run  -m pytest ./tests/unit/test_service/
        ```
    - Run API Layer unit tests using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage run  -m pytest ./tests/unit/test_routes/
        ```
    - Run all the unit tests using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage run  -m pytest ./tests/unit/
        ```
    - 
2. **Integration Tests**
    - Run the integration tests using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage run  -m pytest ./tests/integration/
        ```
3. **Run all the tests**
    - Run all the tests using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage run  -m pytest ./tests/
        ```
4. **View the Coverage Report**
    - View the coverage report using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage report
        ```
    - View the coverage report in HTML format using the following command:
        ```bash
        ~/sentiment-analysis-api$ coverage html
        ```
        - Open the HTML file in the browser:
            ```bash
            ~/sentiment-analysis-api$ firefox ./tests/coverage/coverage_html_report/index.html 
            ```



<!-- GSoC Docs -->
## <img align="center" width="60px" src="https://en.opensuse.org/images/9/91/Gsocsun.png"> GSoC Docs <a id="gsoc"></a>
This repository is part of the [Google Summer of Code (GSoC) 2024](https://summerofcode.withgoogle.com/) program.

- **Contributor:** [Basma Elhoseny](https://github.com/basmaelhoseny01)
- **Mentors:** [Karine](https://github.com/KarinePistili) - [Marc](https://github.com/marcgc21) - [Vinícius](https://github.com/hvini) - [Murilo](https://github.com/murilonND)

### Useful Links:
- GSoC'24 Project Page: [Sentiment Analysis API Project GSoC 24 Program](https://summerofcode.withgoogle.com/programs/2024/projects/V469U9cf)
- Progress Tracking Docs: [GSOC'24 Project Progress and Follow Up Sheet](https://docs.google.com/spreadsheets/d/1wnTACVlsw_JWCWV70Log1DFwilxwmj_azB3TekGA3OY/edit?usp=sharing)
- Meetings Presentations: [Slides](https://drive.google.com/drive/u/1/folders/1SMujWE0p7Xz_CaS0e9qKkLRcDt7YX729)
- Main Reposotory for the Project: [sentiment-analysis-api Repo](https://github.com/ruxailab/sentiment-analysis-api) 
- Integration to [RUXAILAB](https://github.com/ruxailab/RUXAILAB) PR Requests<a id="integration_to_ruxailab"></a>:
    - [PR #533](https://github.com/ruxailab/RUXAILAB/pull/533)
- Wikkis:
    - [Deployment Study Guide](https://github.com/ruxailab/sentiment-analysis-api/wiki/Deployment-Study-Guide)
    - [Speech2Text Tools Survey](https://github.com/ruxailab/sentiment-analysis-api/wiki/Speech2Text-Tools-Survey)
    - [VueJs and Vuetify Study Guide (RUXAILAB Repo)](https://github.com/ruxailab/RUXAILAB/wiki/VueJs-and-Vuetify-Study-Guide)
  
<!-- License -->
## <img align="center" height="60px" src="https://cdn-icons-png.freepik.com/512/1046/1046441.png"> License <a id="license"></a>
This software is licensed under the MIT License. See the [LICENSE](https://github.com/ruxailab/sentiment-analysis-api/blob/main/LICENSE) file for more information.
<div align="center">
    © 2024 RUXAILAB.
</div>
