package pt.candor;

import java.io.File;
import java.io.IOException;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	
	private static final String CSV_FILE ="C:\\Users\\ricar\\eclipse-workspace\\Weka\\resources\\csv\\candor.csv";
	private static final String ARFF_FILE ="C:\\Users\\ricar\\eclipse-workspace\\Weka\\resources\\arff\\candor.arff";
	
	public static void main(String[] args) throws Exception {
		convertCsvToArff(CSV_FILE, ARFF_FILE);
		DataSource source = new DataSource(ARFF_FILE);
		
		//get instances object 
		Instances data = source.getDataSet();
		// new instance of clusterer
		SimpleKMeans model = new SimpleKMeans();//Simple EM (expectation maximisation)
		//number of clusters
		model.setNumClusters(10);
		//set distance function
		//model.setDistanceFunction(new weka.core.ManhattanDistance());
		// build the clusterer
		model.buildClusterer(data);
		System.out.println(model);
	}
	
	public static void convertCsvToArff(String csvFile, String arffFile) throws IOException {
		// load CSV
	    CSVLoader loader = new CSVLoader();
	    loader.setSource(new File(csvFile));
	    Instances data = loader.getDataSet();//get instances object

	    // save ARFF
	    ArffSaver saver = new ArffSaver();
	    saver.setInstances(data);//set the dataset we want to convert
	    //and save as ARFF
	    saver.setFile(new File(arffFile));
	    saver.writeBatch();
	}
}
