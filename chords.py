#!/usr/bin/python

from collections import defaultdict
import music21
import random
import math
import sys


def sampleMultinomial(candidates, weights):
    '''
        return a subset of candidates based on a random number
    '''
    if not candidates or not weights :
        raise Exception("multinomial passed null")

    point = random.random()
    total = 0.0
    for i in range(0, len(candidates)):
        total += weights[i]
        if total > point:
            return candidates[i]
    print "Error in sampling"


def train(ngrams, vocab, durations, note_durations):
    # get list of paths for all midi files you might want
    # songs = corpus.getBachChorales()
    songs = music21.corpus.getComposer('Handel')
    for song in songs:
        print("Learning {0}".format(song))
        parseFile(song, ngrams, vocab, durations, note_durations)


def parseFile(filename, ngrams, vocab, durations, note_durations):
    # parse through the given file and adjust our params accordingly
    key = None
    mode = None
    score = music21.corpus.parse(filename)

    # train durations on the notes, not the chords
    trainDurations(score, durations, note_durations)

    # next train chord ngrams on the chords of the piece
    chords = score.chordify().flat

    oldChords = ['<BOS>'] * N1

    for c in chords:
        if "Chord" in c.classes:
            root = c.root()
            relativeroot = str(music21.interval.notesToChromatic(key, root).semitones)

            oldChords.pop(0)
            oldChords.append(relativeroot)

            ngram = '\n'.join(oldChords)
            ngrams[ngram] += 1

            relativePitches = []
            for p in c.pitchNames:
                relativePitches.append(str(music21.interval.notesToChromatic(key, music21.note.Note(p)).semitones))

            notes = " ".join(relativePitches)+'\t'+mode+'\t'+relativeroot

            vocab.add(notes)

        elif "KeySignature" in c.classes:
            key, mode = c.pitchAndMode
            if mode is None:
                mode = "unknown"
            keySignature = key.name + '\t' + mode
            keys[keySignature] += 1


def trainDurations(score, durations, note_durations):
    '''
    takes in a score object, durations and a vocab
    '''
    oldDurations = ['<BOS>'] * N2

    for part in score.parts:
        for measure in part:
            if type(measure) == music21.stream.Measure:
                for component in measure:
                    if type(component) == music21.note.Note:
                        oldDurations.pop(0)
                        oldDurations.append(component.duration.type)

                        durgram = '\n'.join(oldDurations)
                        durations[durgram] += 1

                        note_durations.add(component.duration.type)


def sampleKey():
    candidates = []
    weights = []

    for k in keys:
        candidates.append(k)
        weights.append(keys[k])

    Z = sum(weights)
    weights = [x / Z for x in weights]

    return sampleMultinomial(candidates, weights)


def sampleDuration(durations, note_durations, quota, oldDurations):
    candidates = []
    weights = []

    for dur in note_durations:
        qlength = music21.duration.convertTypeToQuarterLength(dur)

        if (dur != "complex") and (0 < qlength <= quota) and (quota % qlength == 0):
            candidates.append(dur)
            durgram = '\n'.join(oldDurations) + '\n' + dur
            weights.append(durations[durgram])

    Z = sum(weights) if sum(weights) > 0 else 1
    # divide each weight by the avg. to normalize.
    weights = [x / Z for x in weights]

    return sampleMultinomial(candidates, weights)


def samplePitches(ngrams, vocab, oldChords, mode):
    candidates = []
    weights = []

    for c in vocab:
        if c.split('\t')[1] == mode:
            root = c.split('\t')[2]
            candidates.append(c)
            ngram = '\n'.join(oldChords) + '\n' + root
            weights.append(ngrams[ngram])

    if len(candidates) == 0:
        for c in vocab:
            candidates.append(c)
            weights.append(1.0)

    Z = sum(weights)
    weights = [x / Z for x in weights]

    return sampleMultinomial(candidates, weights)


def writeMelody(p, durations, note_durations, key, interv):
    melody = music21.stream.Part()

    oldDurations = ["<BOS>"] * (N2 - 1)

    previous1 = music21.note.Note(key)
    previous2 = music21.note.Note(key)

    for hmeasure in p:
        hchord = hmeasure[0]
        pitches = hchord.pitches
        m = music21.stream.Measure()

        quota = 4.0

        while quota > 0.0:
            # pick a duration
            dur = sampleDuration(durations, note_durations, quota, oldDurations)
            quota -= music21.duration.convertTypeToQuarterLength(dur)

            # pick a pitch from the chord
            options = []
            for pitch in pitches:
                if not (previous2.name == pitch.name or previous1.name == pitch.name):
                    options.append(pitch)

            if len(options) == 0:
                for pitch in pitches:
                    options.append(music21.interval.Interval(-12).transposePitch(pitch, maxAccidental=1))
                    options.append(music21.interval.Interval(12).transposePitch(pitch, maxAccidental=1))

            penalty = []
            for pitch in options:
                penalty.append(math.fabs(float(music21.interval.notesToChromatic(previous2, music21.note.Note(pitch)).semitones)))

            mx = max(penalty)
            weights = [mx - x + 1 for x in penalty]
            Z = sum(weights)
            weights = [x / Z for x in weights]

            pitch = sampleMultinomial(options, weights)
            n = music21.note.Note(pitch)

            # transpose for audibility
            n = n.transpose(interv)

            # add that note to measure
            n.duration.type = dur
            m.append(n)

            previous2 = previous1
            previous1 = n

        melody.append(m)

    return melody


def sampleRhythm(durations, note_durations, quota):
    oldDurations = ['<BOS>'] * (N2 - 1)
    pattern = []
    quota = 4.0

    while quota > 0.0:
        dur = sampleDuration(durations, note_durations, quota, oldDurations)
        quota -= music21.duration.convertTypeToQuarterLength(dur)
        pattern.append(dur)

    return pattern


def writeSong(ngrams, vocab, durations, note_durations, timeSig):
    keySignature = sampleKey()
    key, mode = keySignature.split()

    s = music21.stream.Score()
    p = music21.stream.Part()

    # sample a rhythmic pattern
    pattern = sampleRhythm(durations, note_durations, 4.0)

    # first sample a harmonic baseline of one chord per measure
    oldChords = ['<BOS>'] * (N1 - 1)

    for i in range(0, 100):
        m = music21.stream.Measure()
        c = samplePitches(ngrams, vocab, oldChords, mode)

        oldChords.pop(0)
        oldChords.append(c.split('\t')[2])

        # append the sampled chord to streams
        absoluteNotes = [music21.note.Note(key).transpose(int(j) - 12) for j in c.split('\t')[0].split()]
        toAdd = music21.chord.Chord(absoluteNotes)
        toAdd.duration.type = "whole"

        m.append(toAdd)

        '''
        full = True
        for dur in pattern:
            toAdd = chord.Chord(absoluteNotes)
            if not full:
                toAdd = toAdd.root()
                full = False
            toAdd.duration.type = dur
            m.append(toAdd)
        '''

        # with some probability, modulate to the dominant
        # corresponding minor/major
        if random.random() > 0.8:
            key = music21.interval.Interval('p5').transposePitch(music21.pitch.Pitch(key), maxAccidental=1).name
        elif random.random() < 0.2:
            if mode == "major":
                key = music21.interval.Interval('M3').transposePitch(music21.pitch.Pitch(key), maxAccidental=1).name
                mode = "minor"
            else:
                key = music21.interval.Interval('m3').transposePitch(music21.pitch.Pitch(key), maxAccidental=1).name
                mode = "major"

        # every 8 measures, new rhythm
        if (i % 4) == 0:
            pattern = sampleRhythm(durations, note_durations, 4.0)

        p.append(m)

    s.append(p)

    # now sample a melody using the chords established
    melody = writeMelody(p, durations, note_durations, key, 24)

    s.append(melody)

    return s


# the N in n-gram
N1 = 2
N2 = 4

# the smoothing factor
# think of this as a measure of originality
lamb = 0.1

# modes that we see
keys = defaultdict(lambda: 0.0)


if __name__ == "__main__":
    # stores the count for every ngram with added lambda smoothing
    ngrams = defaultdict(lambda: lamb)
    # stores the count for every rhythmic ngram with added lambda smoothing
    durations = defaultdict(lambda: lamb)
    # stores the full vocabulary of notes we've seen
    vocab = set()
    # stores all possible note durations
    note_durations = set()

    # train the ngrams and vocab
    train(ngrams, vocab, durations, note_durations)

    # write a short song
    print "Writing a short song"

    s = writeSong(ngrams, vocab, durations, note_durations, 4.0)
    print "Done writing song"

    # write output to MIDI file
    print "Writing output to MIDI file"
    outfile = music21.midi.translate.streamToMidiFile(s)
    outfile.open("output.mid", 'wb')
    outfile.write()
    outfile.close()



    print("\nAnd we're done")
