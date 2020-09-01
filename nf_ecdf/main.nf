// These are defaults which can be overwriten with --output_folder
params.output_folder = "s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/mira/results"
params.batchfile = "s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/mira/dill_manifest.csv"
params.ref = "s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/human_T_beta_bitanova_unique_clones_sampled_1220K.csv"

ref_file = file(params.ref)

Channel.from(file(params.batchfile))
    .splitCsv(header: true, sep: ",")
    .map { sample ->[sample.file, sample.name, file(sample.complete)] }
    .set{ input_channel }

process simple {

    container 'quay.io/afioregartland/python_container'

    publishDir params.output_folder, mode: 'copy', overwrite: true, pattern: '*.feather'
    
    memory '1 GB'
    
    cpus 2

    errorStrategy 'finish'
    
    input: 
        set file, name, file(complete) from input_channel
        file reference from ref_file
    
    script:
    """
    conda activate py36

    pip install git+git://github.com/kmayerb/tcrdist3.git

    curl -k -L https://github.com/agartland/ncov_epitopes/mira_enrichment_compute_ecdf.py -o mira_enrichment_compute_ecdf.py

    # aws s3 cp s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/human_T_beta_bitanova_unique_clones_sampled_1220K.csv ./

    python mira_enrichment_compute_ecdf.py --dill ${complete} \
                                           --ref ${reference}
                                           --ncpus 2 --subsample 100

    # aws s3 cp ./ s3://fh-pi-gilbert-p/agartlan/ncov_tcrs/ --recursive --exclude "*" --include "*.feather"                                       
    """
}
