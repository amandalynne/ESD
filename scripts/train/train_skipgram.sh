mkdir -p vectors/skipgram
cd vectors/skipgram

# Path to semanticvectors jar file. You may need to change this!
PATH_TO_JAR_FILE=~/semanticvectors/target/semanticvectors-5.9.jar

# Corpus (full sentences)
PATH_TO_CORPUS=../../data/corpus/medline_fullsents_v7.txt

# Build positional_index
java -cp $PATH_TO_JAR_FILE pitt.search.lucene.IndexFlatFilePositions $PATH_TO_CORPUS 

# Train 
java -Xmx28G -cp $PATH_TO_JAR_FILE pitt.search.semanticvectors.BuildPositionalIndex -dimension 250 -seedlength 250  -encodingmethod embeddings  -exactwindowpositions -samplingthreshold 0.00001 -trainingcycles 4 -minfrequency 5 -filteroutnumbers -positionalmethod basic -windowradius 2  -docindexing none -luceneindexpath positional_index -numthreads 60

