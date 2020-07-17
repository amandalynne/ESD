mkdir -p vectors/esd
cd vectors/esd

# Path to semanticvectors jar file. You may need to change this!
PATH_TO_JAR_FILE=~/semanticvectors/target/semanticvectors-5.9.jar

# Corpus (concept-path-concept triples)
PATH_TO_CORPUS=../../data/corpus/medline_triples_v7.txt

# Build predication_index
java -cp $PATH_TO_JAR_FILE pitt.search.lucene.LuceneIndexFromSemrepTriples $PATH_TO_CORPUS 

# Train
java -Xmx260G -cp $PATH_TO_JAR_FILE pitt.search.semanticvectors.ESP -luceneindexpath predication_index -minfrequency 5 -mutablepredicatevectors -samplingthreshold .00001 -dimension 8000 -seedlength 4000 -trainingcycles 4 -numthreads 60 -vectortype binary -maxfrequency 1000000 -elementalmethod contenthash
