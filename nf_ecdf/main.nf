// These are defaults which can be overwriten with --output_folder
params.output_folder = "s3://fh-pi-kublin-j-microbiome/training/results"
params.batchfile = "s3://fh-pi-kublin-j-microbiome/training/manifest.csv"


Channel.from(file(params.batchfile))
    .splitCsv(header: true, sep: ",")
    .map { sample ->[sample.name, file(sample.filename)] }
    .set{ input_channel }


process simple {

    container 'quay.io/kmayerb/tcrdist3:0.1.4'

    publishDir params.output_folder, mode: 'copy', overwrite: true
    
    memory '1 GB'
    
    cpus 1

    errorStrategy 'finish'
    
    input: 
    	set name, file(filename) from input_channel
    
    output: 
    	file("${filename}.outfile.csv") into output_channel

    script:
    """
    pip install requests
    hello.py ${filename}
    """
}