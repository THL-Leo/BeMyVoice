## Vertical
We are doing the Accessibility Vertical for Socal Tech Week 2024. We aimed to design a program that promotes inclusivity for everyone.

## Inspiration
When we saw the Accessibility track, we knew we could make a real difference. Our goal with Be My Voice is to break down barriers for those with speech disabilities, making communication easier and more inclusive. By providing real-time ASL translation, we’re not just helping people connect, but empowering them to pursue opportunities, build relationships, and share their voices with the world.

## What it does
Be My Voice has two main features: real-time ASL translation and ASL flashcards for people to learn. We are proud of the model that we trained and utilized OpenAI's ChatGPT to facilitate and improve the output from our model. The flashcards allow people to learn American Sign Language as they go - featuring simple yet universal vocabularies.

## How we built it
The tech stack we chose is JavaScript for the Frontend, Flask for the Middleware API, and Python for the Backend computations. We utilized Python's OpenCV, mediapipe, and Tensorflow library to train our model with over hundreds of photos symbolizing ASL vocabularies. We also borrowed the vast usefulness of ChatGPT 3.5 Turbo model to help us expand on the meaning of these vocabs. This allows the user to better understand what each hand gesture mean in many contexts. We utilized Flask's simpleness to help us quickly connect our Frontend with our Backend, bridging the heavy Computer Vision computation with our laptop cameras. For our UI, we opted for Bootstrap for its wide variety of customization for our interfaces.

## Challenges we ran into
Initially, we wanted to build a React Native app that can be used on iOS, Android, and the web browser. However, we quickly realized that given the time constraint, it will be really difficult to polish all three devices. In addition, finding useful dataset to help train our model proved to be very difficult as most of the dataset came in a variety of forms. As a result, we created our own dataset while learning American Sign Language on the go. We took hundreds of photos and bricked one of our laptops multiple times during training.

## Accomplishments that we're proud of
Given the time constraint, we are proud that we were able to train our model with an acceptable accuracy. There are a lot of hand gestures in American Sign Language and many of which are non-static, meaning they need to be represented as time series. We are also very proud of our team quickly put together a tech stack that offers many options to choose from. Both JavaScript and Python offers immense support for whatever idea we wanted to do.

## What we learned
Before today, none of us have ever used the American Sign Language. We learned that it is much much more difficult than we all imagined, especially for words that are trendy or niche. We also learned that training a non-static time series ASL model proved to be too computation intensive with the hardware we have on our hands. Even after narrowing our datasets down, training the model is still very difficult.

## What's next for Be My Voice
Be My Voice has inspired us to learn more about American Sign Language and Computer Vision. As we finish our semester here at USC, we hope to be able to apply our skills and our interest in improving people's lives and the future together. Fight on!

## Team members (alphabetical order)
Akash Singh, Leo Lee, Reetvik Chatterjee, Sujeet Jawale
