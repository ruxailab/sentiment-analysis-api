# <img align="center" width="60px" src="https://media.licdn.com/dms/image/C5612AQG0_zmxHXI5fQ/article-cover_image-shrink_720_1280/0/1520134887225?e=2147483647&v=beta&t=JswxDuKKsAJdkMfqcRMh0WwZF3wztqx4z-aPDPEt1YM"> Sentiment Analysis API
<div align="center">
    <img src="https://t4.ftcdn.net/jpg/08/92/22/07/360_F_892220774_JALjFtTUwSQBsgd1dOUvagKlhPYrDTKU.jpg"/>
</div>

## <img align="center" width="80px" src="https://cdn.dribbble.com/users/35253/screenshots/3984334/ideate_dribb.gif"> Table of Contents
- <a href="#Overview">Overview</a>
- <a href="#tools">Built Using</a>
- <a href="#demo">Demo</a>
- <a href="#documentaion">API Documenation</a>
- <a href="#started">Get Started</a>
    - <a href="#started">Native</a>
    - <a href="#started">Docker</a>
- <a href="#gsoc">GSoC Docs</a>
- <a href="#license">License</a>

<!-- Overview -->
## <img align="center" width="70px" src="https://cdn.dribbble.com/users/1501052/screenshots/4545496/media/13e279b5c3bd2e8f79067239da3d8633.gif"> Overview <a id="Overview"></a>


<!-- Tools -->
## <img  align= center width =60px  height =70px src="https://media4.giphy.com/media/ux6vPam8BubuCxbW20/giphy.gif?cid=6c09b952gi267xsujaqufpqwuzeqhbi88q0ohj83jwv6dpls&ep=v1_stickers_related&rid=giphy.gif&ct=s"> Built Using<a id="tools"></a>
<table style="border-collapse: collapse; border: none;">
  <tr>
    <td><img height="60" src="https://dragonz.dev/assets/images/os/linux.png"/></td>
    <td><img height="60" src="https://miro.medium.com/v2/resize:fit:1400/0*adyeTInZ7lebNANK.png"/></td>
    <td><img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/2560px-PyTorch_logo_black.svg.png"/></td>
    <td><img height="70" src="https://cdn.worldvectorlogo.com/logos/docker.svg"/></td>
    <td><img height="60" src="https://miro.medium.com/v2/resize:fit:438/1*dQvABiWzbE28OTPYjzElKw.png"/></td>
    <td><img height="80" src="https://miro.medium.com/v2/resize:fit:800/1*HTYDFA422w071hjbMuMqGA.png"/></td>
  </tr>
</table>

<!-- API Documentation -->
## <img align="center" width="80px" src="https://cdn.dribbble.com/users/1874602/screenshots/5647628/send-icon.gif"> API Documentation <a id="documentaion"></a>
You can access the API documentation at [http://localhost:8000/apidocs](http://localhost:8000/apidocs) after running the Flask App.

<!-- Postman -->
### Postman Collection
For testing the API endpoints, you can use the following Postman collection:
- [RuxAiLab Sentiment Analysis APIs Postman Collection](https://www.postman.com/interstellar-shadow-582340/workspace/ruxaailab/collection/31975349-d17198fa-8c4f-41d4-9870-1dc6e7443bc3) 
 
<!-- Demo -->
## <img align="center" width="80px" src="https://cdn.dribbble.com/users/346181/screenshots/2332299/rf-icon-_premium-quality-videos_.gif"> Demo <a id="demo"></a>

<!-- Getting Started -->
## <img align="center" width="60px" height="60px" src="https://media3.giphy.com/media/wuZWV7keWqi2jJGzdB/giphy.gif?cid=6c09b952wp4ev7jtywg3j6tt7ec7vr3piiwql2vhrlsgydyz&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=s"> Get Started <a id="started"></a>

<!-- Native -->
### <img align="center" height="40px" width="50px" src="https://media.lordicon.com/icons/wired/outline/743-web-code.gif"> Native <a id="native"></a>
1. **Clone the Repository**
    ```bash
    git clone https://github.com/ruxailab/sentiment-analysis-api.git
    cd sentiment-analysis-api/app
    ```

2. **Create the Virtual Environment**
    ```bash
    python3 -m venv env
    ```
    
3. **Activate the virtual environment**
    ```bash
    source env/bin/activate
    ```

4. **Install Dependecies**
    ```bash
    pip install -r requirements.txt
    ```
    
5. **Run Flask App**
   -    In Debug Mode  [port 8000]
        ```bash
        python app.py
        ```
        
6. **Run API Documentation**
     - Access the API documentation at: http://localhost:8000/apidocs
   
<!-- Docker -->
### <img align="center" width="60px" src="https://miro.medium.com/v2/resize:fit:1400/1*wXtyhpOL5NK_w39UvZpADQ.gif"> Docker <a id="docker"></a>


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
- Integration to [RUXAILAB](https://github.com/ruxailab/RUXAILAB) PR Requests:
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
