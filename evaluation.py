#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:47:17 2019

@author: fwang
"""

# apply perplexity to evaluate different models

# read the model results
import pandas as pd
results = pd.read_csv("results_for_evaluation.csv")

# perplexity part
import collections, nltk

# use results from Vanilla_MLE, Vanilla_Sample and REINFORCE to form corpus
corpus = """
I'm not a doctor.
sammy wrote the test sammy wrote the test.
I'm going to the movies.
you're the only one who knows what's going on.
he's got a big mouth.
they're not the only ones who have been in the system.
I don't know who he is.
I'm sure the dag will be here soon.
I'm not a beggar.
I was born in the world of the world
I'm just a little bit of a little bit of a little bit of a little bit of a
he's not afraid of the dark.
I think it's the same thing.
I'll be back in a few days.
you know what I mean?
I'm gonna be a little late for the movie.
she's not going to be a party member.
I don't think so.
I'll be back in a minute.
I was in the middle of the road.
I'm gonna be in the kitchen.
he's not gonna let them go through the motions.
you have to go back to the city.
the war is not over.
I don't think so.
he was in the middle of the night.
we're gonna be late for the meeting.
I'm gonna get'em.
I'll be right back
well everything you did was totally untrue.
objects pick objects up objects objects objects objects
get him outta here first!
when they are conquered and you surrender they will control all of us.
I'm sure he's unhappy with the fact.
it's the new priority only.
the church … suffered the winds of 900 meters with horse aid.
I could at least think.
is this the money eugene?
as in childhood I took you to one tenth of a postcard
shiny gold gold and silver in the mountains
even the program is the only truth.
I'm still very nervous.
you wiii be transferred to the target court
it's full of crime…
you mean you're living?
the timing's good and bad.
and it requires time to be wrong and one believes it.
let's go for some fun.
my body was just pounding.
was this gay or red?
they asked him whether we could go see ernesto?
the one is a multiple choice.
there are people who hope that in tokyo they won't risk lossing their departing.
is it possible as old as you?
he should have been somewhere out here.
c'mon c'mon let's get outta here.
those weak rotten guys had me in a rush on a broken swoop.
sure if you don't put it into the water
I don't know how long it's been.
I'm not sure that's relevant.
I'm going to get you.
I'm afraid I'm not ready yet.
he's a little big of a pain.
they're not the only ones who can help us.
he was a good man.
I don't think so.
I don't know what to do.
and the world was full of joy
I'm not gonna be a baby.
he's also not afraid of me.
maybe it's just a coincidence.
I'm not sure if you can handle it without bothering them.
I don't know what you're talking about.
I'm gonna get you a new one.
we'll have to get her back.
it's a genatic disorder.
I'll drive you home.
I was just trying to help you.
I don't want to be a vegetable.
I don't know what to do.
I think I have to.
and the people who are not the same.
I'm not going to be a burden on you.
he's not a good guy.
I'll be right back.
I don't know what to do.
I'm gonna be a little busy with the dishes.
A few month, I guess.
so he took the pills
I'm going to get you a new car.
I'm afraid of your power.
I don't think so.
can we find someone else?
he was a very good man.
I don't think so.
how much money you owe me?
and the world was full of joy
I'm not going to be a father.
he's such a brave man!
I don't know if it's a coincidence.
I'm not going to be a nuisance
I don't want to be a girl.
I'm gonna get you a new tv set.
I don't think we have to worry about her.
I don't think so.
I don't have a car.
I was just trying to get a little help.
I'm not a very good cook.
he's not going to be here any longer.
I think I have to.
the war is still going on.
I don't want to be a burden.
he's got a lot of nerve.
we're gonna have to get to the station.
I was just a little bit nervous.
I'm gonna make you a little dinner.
a couple of weeks .
the body ?
I'm leaving.
no.
beside me … ?
I'm not here.
what about the money?
do you stay it?
we're going to get a long time. We're going sightseeing.
_UNK …
what ?
you're a fool.
no. no no no  no no no no no no no no no no
five.
I'm not a bodyguard of it.
there's another sick damage?
what?
considering considering what me considering listen scooter gump it lotta generous cereal harder between lombard
what about her ?
what ?
no.
nope.
i ' m not going to sleep .
i ' m sorry .
i ' m sorry .
i ' m not telling him .
i ' m sorry .
what ?
a little bit of a dozen .
"""

tokens = nltk.word_tokenize(corpus)

def unigram(tokens):    
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model [f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model

def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

model = unigram(tokens)
# compute perplexity for Vanilla MLE Method
perplexity_vanilla_mle = []
for testset in results['Vanilla_MLE']:
    perplexity_vanilla_mle.append(perplexity(testset,model))
perplexity_vanilla_mle_pd = pd.DataFrame(perplexity_vanilla_mle).rename(columns={0: 'perplexity'})
perplexity_vanilla_mle_pd.mean()

# compute perplexity for Vanilla Sample Method
perplexity_vanilla_sample = []
for testset in results['Vanilla_Sample']:
    perplexity_vanilla_sample.append(perplexity(testset,model))
perplexity_vanilla_sample_pd = pd.DataFrame(perplexity_vanilla_sample).rename(columns={0: 'perplexity'})
perplexity_vanilla_sample_pd.mean()

# compute perplexity for REINFORCE Method
perplexity_reinforce = []
for testset in results['REINFORCE']:
    perplexity_reinforce.append(perplexity(testset,model))
perplexity_reinforce_pd = pd.DataFrame(perplexity_reinforce).rename(columns={0: 'perplexity'})
perplexity_reinforce_pd.mean()

# compute perplexity for REINFORCE-Monte-Carlo Method
perplexity_regs_mc = []
for testset in results['REGS_Monte_Carlo']:
    perplexity_regs_mc.append(perplexity(testset,model))
perplexity_regs_mc_pd = pd.DataFrame(perplexity_regs_mc).rename(columns={0: 'perplexity'})
perplexity_regs_mc_pd.mean()

# compute perplexity for Our Method
perplexity_ours = []
for testset in results['Ours']:
    perplexity_ours.append(perplexity(testset,model))
perplexity_ours_pd = pd.DataFrame(perplexity_ours).rename(columns={0: 'perplexity'})
perplexity_ours_pd.mean()

