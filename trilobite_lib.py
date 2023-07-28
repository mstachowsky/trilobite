import xml.etree.ElementTree as ET
import os
import sys
from transformers import AutoTokenizer
import transformers
import torch
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import traceback

def create_or_clean_subdirectory(root_directory, subdirectory_name):
    # Construct the full path of the subdirectory
    subdirectory_path = os.path.join(root_directory, subdirectory_name)

    if os.path.exists(subdirectory_path):
        # Subdirectory exists, so delete everything in it
        for item in os.listdir(subdirectory_path):
            item_path = os.path.join(subdirectory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove files
            elif os.path.isdir(item_path):
                os.rmdir(item_path)  # Remove directories
    else:
        # Subdirectory doesn't exist, so create it
        os.makedirs(subdirectory_path)

def setup_directories(root):
    root_dir = root.find("rootdir").text.strip()
    print("TODO: This automatically deletes everything, so be careful. Will fix this later.")
    create_or_clean_subdirectory(root_dir,"extract")
    create_or_clean_subdirectory(root_dir,"summarize")
    create_or_clean_subdirectory(root_dir,"finalize")
    create_or_clean_subdirectory(root_dir,"evaluate")
    
    """
    cont = 'd'
    if(os.path.exists(root_dir)):
        print("Path already exists. Do you want to delete and recreate the directory structure, or continue as it is?")
        cont = input("Input 'd' to delete or 'c' to continue without deleting: >> ")
    if cont == 'd':    
        create_or_clean_subdirectory(root_dir,"extract")
        create_or_clean_subdirectory(root_dir,"summarize")
        create_or_clean_subdirectory(root_dir,"finalize")
        create_or_clean_subdirectory(root_dir,"evaluate")
    """

#Extracts all of the queries from the root of the tree
def extract_queries(root):
    queries = []
    for analysis in root.findall(".//analysis"):
        query_element = analysis.find("query")
        if query_element is not None and query_element.text:
            queries.append(query_element.text.strip())
    return queries

#Extract a list of dicts. Note that this list has a 1:1 index correspondence with the list of queries
def extract_rubrics(root):
    # Initialize a list to store the dictionaries for rubrics
    rubrics_list = []

    # Extract rubric details from each "analysis" element
    for analysis in root.findall(".//analysis"):
        # Initialize a dictionary to store rubric information for each analysis
        rubric_dict = {}

        # Extract rubric details from each "category" element under the current "analysis"
        for category in analysis.findall(".//category"):
            rating_element = category.find("rating")
            criteria_element = category.find("criteria")

            if rating_element is not None and criteria_element is not None:
                rating = rating_element.text.strip()
                criteria = criteria_element.text.strip()

                # Add the rubric details to the rubric_dict for this analysis
                rubric_dict[rating] = criteria

        # Append the rubric_dict to the rubrics_list for this analysis
        rubrics_list.append(rubric_dict)

    # Return the list of dictionaries for rubrics
    return rubrics_list

def create_extract_queries(root):
    queries = extract_queries(root)
    extract_query_list = []
    print(root.find(".//preamble_extract"))
    if root.find(".//preamble_extract") is not None:
        preamble_extract = root.find(".//preamble_extract").text.strip()
        start_string = "<s>[INST] <<SYS>>" + preamble_extract
        end_string = """ The text is:
        <</SYS>>
        """
        
        qu = start_string + " " + end_string
        extract_query_list.append(qu)
        print(qu)
        """
        for query in queries:
            qu = start_string + " " + query + " " + end_string
            extract_query_list.append(qu)
        """
    return extract_query_list


def create_summarize_queries(root):
    queries = extract_queries(root)
    summary_query_list = []
    if root.find(".//preamble_summarize")is not None:
        preamble_summarize = root.find(".//preamble_summarize").text.strip()
        start_string = "<s>[INST] <<SYS>>" + preamble_summarize
        end_string = """The text is: 
        <</SYS>>
        """
        qu = start_string + " " + end_string
        summary_query_list.append(qu)
        
        """
        for query in queries:
            qu = start_string + " " + query + " " + end_string
            summary_query_list.append(qu)
        """
    return summary_query_list
    

def create_finalize_queries(root):
    queries = extract_queries(root)
    finalize_query_list = []
    if root.find(".//preamble_finalize")is not None:
        preamble_finalize = root.find(".//preamble_finalize").text.strip()
        start_string = "<s>[INST] <<SYS>>" + preamble_finalize
        end_string = """ The text is: 
        <</SYS>>
        """
        qu = start_string + + " " + end_string
        finalize_query_list.append(qu)
        """
        for query in queries:
            qu = start_string + " " + query + " " + end_string
            finalize_query_list.append(qu)
        """
    return finalize_query_list
    
def create_evaluate_queries(root):
    queries = extract_queries(root)
    rubrics = extract_rubrics(root)

    rubric_query_list = []
    if root.find(".//preamble_evaluate")is not None:
        preamble_finalize = root.find(".//preamble_evaluate").text.strip()
        start_string = "<s>[INST] <<SYS>>" + preamble_finalize
        mid_string = """The following arer the classes in which you should classify the student work:
        
        """

        for i,query in enumerate(queries):
            qu = start_string + " " + query + " " + mid_string
            
            for key in rubrics[i].keys():
                qu = qu + key + ": " + rubrics[i][key] + "\n"
            
            qu = qu + "\n Output one of the above " + str(len(rubrics[i])) + """ Classes, and briefly provide feedback. The text is: 
            
            <</SYS>>
            """
            
            rubric_query_list.append(qu)
    return rubric_query_list

"""
Processes pdf or docx files only. Not .txt. Extract a single page, then analyze it, and output your analysis, repeat for all pages
"""
def extract_and_analyze_from_docs(input_filename,pipeline,tokenizer,output_filename=None,sourceDir='.',destinationDir='.',query=""):
    #load the document
    print("Extract: ", os.path.join(sourceDir,input_filename))
    loader = PyPDFLoader(os.path.join(sourceDir,input_filename))
    txt_docs = loader.load_and_split()
    
    #load_and_split() #split on page
    if txt_docs != []:
        analysis = ""
        for page_num, page in enumerate(txt_docs):
            txt = page.page_content
            prompt = query + txt + " \n Provide your analysis now. [/INST]"
            sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=768
            )
            for seq in sequences:
                model_out = seq['generated_text'].split("[/INST]")[1].strip()
                analysis = analysis + "=====PAGE " + str(page_num) + "\n" + model_out + "\n\n"

            print("Processed page......", str(page_num))
        txtname = input_filename.split(".")[0] + "_processed.txt"
        if output_filename is not None:   
            txtname = output_filename
        with open(os.path.join(destinationDir,txtname),'w',encoding='utf-8',errors='ignore') as writer:
            writer.write(analysis)
        print("Done extraction")
    else:
        print("Error with ", input_filename, '\n')


"""
Analyze .txt documents. This one works on the whole document up to chunk sizes.
"""
def extract_and_analyze_from_text(input_filename,pipeline,tokenizer,output_filename=None,sourceDir='.',destinationDir='.',query="",chunk_size = 14000):
    #load the document
    print(os.path.join(sourceDir,input_filename))
    #loader = TextLoader(os.path.join(sourceDir,input_filename))
    raw_txt = ""
    with open(os.path.join(sourceDir,input_filename),'r',encoding='utf-8',errors='ignore') as reader:
        raw_txt = reader.read()
    text_splitter = CharacterTextSplitter(        
        chunk_size = chunk_size,
        chunk_overlap  = 0,
        length_function = len,
        )
    
    txt_docs = text_splitter.create_documents([raw_txt])#loader.load_and_split(chunk_size=3000)

    #load_and_split() #split on page
    if txt_docs != []:
        analysis = ""
        for page_num, page in enumerate(txt_docs):
            txt = page.page_content
            prompt = query + txt + " \n Provide your analysis now. [/INST]"
            sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512
            )
            for seq in sequences:
                model_out = seq['generated_text'].split("[/INST]")[1].strip()
                analysis = analysis + "=====PAGE " + str(page_num) + "\n" + model_out + "\n\n"
                #print("========",paper,"\n",model_out,"\n\n")
            print("Processed page......", str(page_num))
        txtname = input_filename.split(".")[0] + "_processed.txt"
        if output_filename is not None:   
            txtname = output_filename
        with open(os.path.join(destinationDir,txtname),'w',encoding='utf-8',errors='ignore') as writer:
            writer.write(analysis)
        
        print("Done")
    else:
        print("Error with ", input_filename, '\n')

#This is an absolute path, but it avoids us having to use a cached file
model_path = "D:\models\llama2_13B_July27_2023\llama2_13b\snapshots\july27"

def trilobite(xml_file,model="meta-llama/Llama-2-13b-chat-hf"):
        
    #Load the XML    
    tree = ET.parse(xml_file)

    #Get the root of the tree
    root = tree.getroot()
    
    #ensure the raw directory exists
    raw = root.find(".//rawdir").text.strip()
    rootdir = root.find(".//rootdir").text.strip()
    if raw is None:
        print("Error: Raw directory does not exist.")
        return
    
    #It exists, so we are good to go
    #create or clean the directories
    setup_directories(root)
    #load the model
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    #extract
    extract_q = create_extract_queries(root)
    print(extract_q)
    eval_dir = "extract"
    if extract_q != []: #TBH, this shouldn't be possible, otherwise all other stuff fails
        for file in os.listdir(raw):
            for query in extract_q:
                try:
                    extract_and_analyze_from_docs(file,pipeline,tokenizer,output_filename=file.split(".")[0] + "_extracted.txt",sourceDir = raw, destinationDir=os.path.join(rootdir,"extract"),query = query)
                except:
                    print("Problem with " + file)
                    traceback.print_exc()
                
    #Summarize
    summarize_q = create_summarize_queries(root)
    if summarize_q != []:
        eval_dir = "summarize"
        for file in os.listdir(os.path.join(rootdir,"extract")):
            for query in summarize_q:
                try:
                    extract_and_analyze_from_text(file,pipeline,tokenizer,output_filename=file.split(".")[0] + "_summarized.txt",sourceDir = os.path.join(rootdir,"extract"), destinationDir=os.path.join(rootdir,"summarize"),query = query)
                except:
                    print("Problem with " + file)
                    traceback.print_exc()
    
    #Finalize
    finalize_q = create_finalize_queries(root)
    if finalize_q != []:
        eval_dir = "finalize"
        for file in os.listdir(os.path.join(rootdir,"summarize")):
            for query in summarize_q:
                try:
                    extract_and_analyze_from_text(file,pipeline,tokenizer,output_filename=file.split(".")[0] + "_finalized.txt",sourceDir = os.path.join(rootdir,"summarize"), destinationDir=os.path.join(rootdir,"finalize"),query = query)
                except:
                    print("Problem with " + file)
                    traceback.print_exc()
                    
    #evaluate
    evaluate_q = create_evaluate_queries(root)
    if evaluate_q != []:
        for file in os.listdir(os.path.join(rootdir,eval_dir)):
            for query in evaluate_q:
                try:
                    extract_and_analyze_from_text(file,pipeline,tokenizer,output_filename=file.split(".")[0] + "_evaluated.txt",sourceDir = os.path.join(rootdir,eval_dir), destinationDir=os.path.join(rootdir,"evaluate"),query = query)
                except:
                    print("Problem with " + file)
                    traceback.print_exc()
    

if len(sys.argv) != 2:
    print("Error! One command line argument is required - the path to the XML file that is used for inference.")
    sys.exit()
xml_path = sys.argv[1]

print("Starting trilobite with input: " + xml_path + "\n")

trilobite(xml_path,model=model_path)