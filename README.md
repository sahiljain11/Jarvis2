# YouTube Video
<a align="center" href="http://www.youtube.com/watch?feature=player_embedded&v=msXS-9NGt0I
" target="_blank"><img align="center" src="http://img.youtube.com/vi/msXS-9NGt0I/0.jpg" 
alt="Jarvis 2.0" width="960" height="720"/></a>

[Devpost Link](https://devpost.com/software/jarvis-2-0-i1b8vo)

# Inspiration

Iron Man’s AI assistant, Jarvis, in the Marvel Cinematic Universe (MCU) was the main inspiration for our product. We wanted to create a way for users to easily access and interact with their favorite applications in one dashboard without having to touch their mouse. We hoped Jarvis 2.0 would increase accessibility to technology, as well as change the way we interact with our machines.

# What it does

Jarvis acts primarily as a personal assistant! Jarvis can be used to access the modern day necessities: emails, music, time, and calendar. Given the current pandemic, we included a COVID-19 widget to inform users about cases and trends in counties and countries. Jarvis allows its users easy access to these necessities through mouseless control.
How we built it

For our hand motion detection, we used a Leap Motion Controller. With UltraLeap’s built in library, we were able to track the coordinate information of various joints, fingers, etc. Taking these coordinates, we created an LSTM model in PyTorch to act as a classifier for various hand gestures.

For the backend of our apps, we used APIs to retrieve data necessary for Jarvis’ operation. Specifically, the APIs obtained data from Spotify, Gmail, Google Calendar and OpenWeatherMap. To create the Corona app, we used Pandas to pull data from .csv files provided by Johns Hopkins, New York Times and Data Package Core Datasets on Github*. Our time app required the Python time library. In addition, we used various Python libraries such as BeautifulSoup and PySpellChecker to create an auto-corrector for the Corona app.

Jarvis’ GUI was made primarily in QML, due to the language’s compatibility with a Python library, PySide2, a Python wrapper for the C++ Library Qt. We used Figma to design all of Jarvis’s buttons and frames. We also used Blender to render a 3D video loop of blue electricity to serve as the background.

Due to the nature of Jarvis, many API requests are needed to initialize Jarvis. To optimize our initialization time, we used the Python threading library.

# Challenges we ran into

Our first major challenge was determining which programming language we should use for the front end. After a few weeks of research, we decided on QML for two reasons. One, QML can run as an app, independent from an internet browser. Secondly, Python and Leap Motion can be easily integrated into QML. Since UltraLeap required Python 2.7 and GUI required Python 3.7, we used a Flask server to act as an intermediary between the two. From a broad perspective, the coordinate information and ML features would be sent to the server via POST request, the server would process and store the necessary gesture and hand position coordinates, and finally the GUI would use a GET request to receive the necessary information.

While implementing our apps, we dealt with lag issues caused by large API requests. In order to decrease the lag in our front end, we optimized our initial requests to retrieve only the data that the user sees, and allowed the user to further choose what specific information they want to see inside of an app. We added threading for the initialization in order to reduce startup times.

The LeapMotion controller itself also had its own set of issues. For example, if I closed my fist, sometimes not all fingers would be placed down and thus throw off the model and therefore the corresponding action as well. This problem tended to magnify as more processes were running. As a result, we reduced our number of gestures to reduce the scope of this issue. Though the problem still remains, it was more manageable to work with.

# Accomplishments that we are proud of

One of our most important steps was actually bringing together individual widgets to create a single user interface. One really big moment that we were happy with was finally creating a dashboard that encapsulated all the widgets. By bringing everything together, we’d finally created one of the most important parts of our minimum viable product, and from there, we could work on editing and adding to our prototype.

Our teams’ collective skills were mostly focused on the backend. Another particularly great moment was creating and implementing the electricity video (the animated dashboard background) using Blender.

Finally, another of our favorite moments was when the LeapMotion controller manipulated the GUI successfully the first time. At the time, the model was a bit shaky, and though the interface was coming together, the hand tracking portion had some eccentric behavior. Nevertheless, this culminated into a better sense of accomplishment when everything was getting put together.

# What we learned

Going into SummerHacks, several members of the Jarvis 2.0 team were fairly new to coding, so there was a significant learning curve. As a result, a lot was learned. First, several APIs were implemented into the creation of the widgets on the user interface. To do this, we had to learn the APIs and how to handle queries and format the received data. Furthermore, our team’s experience was mostly in the backend. Because we had a display interface, many of us worked seriously in the frontend for the first time. As part of this, we spent a lot of time familiarizing ourselves with Figma for the static elements and Blender for the 3D animations.

Our team spent a lot of time working, both on Jarvis 2.0 and on our individual projects, research, classes, and internships. We wanted to use this SummerHacks to learn and work on a fun project we’d be proud of. In these last few months, we’ve gotten better at balancing our work, and incorporating fun into the process. While we learned a lot of important coding skills, and improved many more, the soft skills we developed were just as important. As a team, we’ve gotten stronger at communicating our needs and our processes, and sustaining team morale. I’m really proud at how we continued to power through such an extended project.

# What's next for Jarvis 2.0?

The broad goal for Jarvis 2.0 is to provide a highly accessible daily aid. First, gesture tracking is an extremely powerful tool, and it would be amazing to be able to incorporate sign language to allow for even greater accessibility in interaction. It would challenge the way we interact with technology in our spaces, using language and movement in place of touch. We would like to consolidate the gesture tracking model and the LeapMotion into a small, portable device separate from the user’s local machine so that the quality of gesture prediction is not dependent on the user’s computational power. Implementing NLP would also further this goal for a number of reasons. It would allow for a truly hands free experience, not only increasing accessibility but also ease of use. Furthermore, integrating an audio visualizer along with NLP would create another dimension for visualizing this function and bringing more life to Jarvis 2.0.

We would also like to expand the number of Jarvis 2.0’s widgets and add extra functionality to the widgets. For example, we could take advantage of the Spotify API’s more advanced features to allow our client to search for songs with a specific beat, feel, or timbre. For the Coronavirus widget, we plan on adding geographical information about the nearest testing centers and (in the future) vaccination facilities. We especially want to incorporate educational tools such as a digital whiteboard and slide presentation widget. With Jarvis 2.0’s hand tracking capabilities, educators would have the freedom to move around the classroom and engage on a more personal and dynamic level with students rather than remain confined to a single and generally distant location.
