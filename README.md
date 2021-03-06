# Glissando


This project generates music. In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sequence of text or speech. The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams are collected from a song corpus using the music21 module.


Admittedly the music is kinda stale.


## How it works


Glissando trains a Bayesian n-gram model by listening to music and remembering important features such as rhythmic patterns and chord progressions. It then generates a new song in that style. First it samples a chord for each measure from the n-gram distribution based on frequently observed chord progressions in its training set.

Once it has done that it samples a melody for that measure. The melody is first given a rhythm, again using an n-gram model trained on the dataset. Once the rhythm is set, it samples the pitches for the melody from the notes that constitute the chord for that measure, weighting them by proximity to the previous note, but excluding the previous note and the one that came before it.


Music21 allows us to parse musical scores stored in either mxml format or midi. The n-gram learner was implemented by hand.

###### note: if you'd like this to run faster you might have better luck with [pypy](http://pypy.org/)

## Challenges

There were some issues with getting the music to sound natural due to the sparsity of the dataset. A certain chord will only appear a few times, so it was important that we figure out how to properly share parameters. Also, every song is written in a different key, so we needed a way to normalize the notes and chords. Finally while we were working on the melody module we ran into some issues with making the progression sound natural. When you deconstruct a melody, it's usually a sequence of notes that appear often in a connected string that move from one chord to the next. Simulating this process was especially difficult since if you use any notes outside of the current chord, it will clash.

## Accomplishments

I'm very proud of the way that we were able to parameterize the model. We found a way to normalize the chords to key signature, and also share parameters for two chords with the same root note that may appear in different octaves or in different inversions (such chords would sound only slightly different to the human ear, and may be used in place of each other without affecting the song).

Also, one big problem with modern computer generated music is that it sounds aimless, and doesn't have much direction. To remedy this, I added a parameter that gives each measure a slight probability of modulating to a different key, or perhaps even changing from the major (uplifting) to the minor (somber) mood. This naturally happens quite often in music, and when the automated song does this it sounds more realistic and interesting.

Finally, I think the end result sounds very nice, and can actually be listened to quite comfortably for a while.


## What's next

The original paper that I read on paintings used a neural network model which can automatically extract and combine interesting features. We weren't able to do that for this Hackathon due to constraints on computing power, but I'd be very interested to see if an approach like that would work in the future.

Another important feature of music is the notion of repeated phrases, melodies, and motifs. Right now the model does not learn these repeated structures beyond the n-grams, but it doesn't have a good sense of the broader structure of the piece and when the melodies should be repeated or explored. This is as of yet a completely unsolved problem in computer generated music--even the neural network approach cannot solve it--but I'd be very eager to see if there's a solution somewhere out there.

## Forked from the fantastic HopHacks project by [asivrat1](http://github.com/asivrat1).
