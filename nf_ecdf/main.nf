// These are defaults which can be overwriten with --output_folder
params.output_folder = "s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/mira/results/"
params.batchfile = "s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/mira/csv_manifest.csv"
params.ref = "s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/human_T_beta_bitanova_unique_clones_sampled_1220K.csv"

ref_file = file(params.ref)

Channel.from(file(params.batchfile))
    .splitCsv(header: true, sep: ",")
    .map { sample ->[sample.file, sample.name, file(sample.complete)] }
    .set{ input_channel }
// .take( 3 )
// .randomSample( 2 )
process mira_ecdf {

    container 'quay.io/afioregartland/python_container'

    publishDir params.output_folder, mode: 'copy', overwrite: true
    
    input: 
        set file, name, file(complete) from input_channel
        file reference from ref_file
    
    output:
        file 'results' into outchan

    script:
    """
    # echo "source /opt/conda/etc/profile.d/conda.sh" >> \$CONDA_ENVIRONMENT
    # echo "conda activate py36" >> \$CONDA_ENVIRONMENT
    # source /opt/conda/etc/profile.d/conda.sh
    # conda activate py36

    conda run -n py36 pip install git+git://github.com/kmayerb/tcrdist3.git

    curl -k -L https://raw.githubusercontent.com/agartland/ncov_epitopes/master/mira_enrichment_compute_ecdf.py -o mira_enrichment_compute_ecdf.py

    # aws s3 cp s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/human_T_beta_bitanova_unique_clones_sampled_1220K.csv ./

    conda run -n py36 python mira_enrichment_compute_ecdf.py --rep ${complete} \
                                           --ref ${reference} \
                                           --ncpus 20 #--subsample 100
    mkdir results
    mv *.feather results/
    aws s3 cp ./results s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/mira/manual_results/ --recursive --exclude "*" --include "*.feather"                                       
    """
}
