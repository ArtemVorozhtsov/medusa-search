import yaml


if __name__=="__main__":
    
    print('MEDUSA search engine setup.\n')
    print('Enter path to MEDUSA package.\n')
    medusa_repository_path = str(input())
    print('Enter path to database with HRMS spectra in .mzXML.\n')
    hrms_database_path = str(input())
    
    # Add medusa_repository_path and hrms_database_path to config.yaml file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['medusa_repository_path'] = medusa_repository_path
    config['hrms_database_path'] = hrms_database_path
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file)
        
    print('Config file changed.\n')
