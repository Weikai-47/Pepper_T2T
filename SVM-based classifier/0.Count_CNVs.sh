fr = open("/data/pepper/CaT2T.CBGs.bed",'r')

REGIONS = []

for line in fr:
    chr = line.strip().split()[0]
    start = line.strip().split()[1]
    end = line.strip().split()[2]

    region = chr + ":" + start + "-" + end

    REGIONS.append(region)

print(REGIONS)


import glob

SAMPLES = [i.split("/")[-2] for i in glob.glob("/data/pepper/*/CRR*_CaT2T.bam")]

print(SAMPLES)
print("Total sample size: ",len(SAMPLES))


rule all:
    input:
        expand("{sample}/{sample}_{region}.CN",sample=SAMPLES,region=REGIONS)


rule amycne:
    input:
        gc = "/data/pepper/gc_content.tab",
        cov = "{sample}/{sample}.regions.bed"
    output:
        "{sample}/{sample}_{region}.CN"
    threads:
        4
    resources:
        mem_mb = 8000
    params:
        "{region}"
    shell:
        """
        python /home/biotools/AMYCNE-master/AMYCNE.py \
        --genotype --gc {input.gc} \
        --coverage  {input.cov} \
        --R {params} > {output}
        """
