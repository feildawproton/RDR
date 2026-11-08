This project will contain a bunch of stuff related to radar processing that i am interested in.

From the outset there will be implentations of:
    - the polar format algorithm using both sync interpolation and incorporating the chirp-z transform in the range dimension
    - the range migration algortirm
    - simulations to support testing of step-chirp radars we don't have access to on the open internet
    - the CLEAN algorithm
    - backprojection (slower than PFA but easier to understand; serves as the cross-check for the other image formers)

We will prolly like to do simulations of dwells, sub-dwells, and CPIs.  but that's later.

Programming languages and whatnot:
    - the image formation algorithms need to be differentiable.  I think pytorch is a good option for this; but, as long as pytorch can use the gradient IDK
    - sarpy and sarkit are frustraitingly slow and I am very interested reading CPHDs faster
    - if CLEAN can be faster in c and cuda then let's read sicds in c (actually i'm not firm on that, really it's just a suggestion)

The ultimate goal:
    - there should be some instalable package that enables:
    - examples to be written in an examples directory, where:
    - the examples are either ipny or interactive GUIs that remind the old human of the algoriths and physics involved without specific or i/o and such

references:
    - there is a references subdir to some standards that should be used
    - if needed reference the goals of ../../diffpfa and ../../CLEAN_SAR, but disregard all code and maybe even some of the goals.  only use these for clues.  they will mislead.

ask for clarity

organiztion
    - very intereseted in src layout that describes the import hierachy.  where possible modules shouldn't import from other modules at the same level 
