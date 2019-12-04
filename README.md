# VISTA
VISTA (Intelligent Seek And Trace of Video) 

Many a time, we come across situations where we want to jump to a specific
frame range, like a certain part of a lecture, parts where only Joker is having
the screen-time and parts where we can enjoy the cool Bat-Mobile. For this,
we usually use a hit and trial method to arrive at the desired frame index.
How cool would it be if we could just enter a text specifying our required
object and the video renderer would display the timelines where the specific
object is present. We could directly jump to the part where there is only Bat-
Mobile or where there’s a phenomenal performance by Joker in a classic
Batman movie! Sleek….

Well, this is what we have implemented in our project!
Our idea is inspired by Google’s latest feature addition to it’s smartphone,
Pixel 4. Google’s ML engine finds specific parts of a speech by taking a user
provided search input, we want to accomplish the same for videos. We have implemented this 
using existing Faster RCNN implementation for detecting relevant object frames, a custom CNN trained on custom superhero dataset for classifying those relevant objects (Person class in this case) to the target superhero and then tracking the detected object using SORT. A figure explaining the project is
provided below: 

