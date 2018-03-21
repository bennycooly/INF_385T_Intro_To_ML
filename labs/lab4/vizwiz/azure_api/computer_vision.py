
import io
import requests
import json
from collections import OrderedDict

vision_base_url = "https://southcentralus.api.cognitive.microsoft.com/vision/v1.0/"
vision_analyze_url = vision_base_url + "analyze"

categories = [
    "abstract_",
    "abstract_net",
    "abstract_nonphoto",
    "abstract_rect",
    "abstract_shape",
    "abstract_texture",
    "animal_",
    "animal_bird",
    "animal_cat",
    "animal_dog",
    "animal_horse",
    "animal_panda",
    "building_",
    "building_arch",
    "building_brickwall",
    "building_church",
    "building_corner",
    "building_doorwindows",
    "building_pillar",
    "building_stair",
    "building_street",
    "dark_",
    "drink_",
    "drink_can",
    "dark_fire",
    "dark_fireworks",
    "sky_object",
    "food_",
    "food_bread",
    "food_fastfood",
    "food_grilled",
    "food_pizza",
    "indoor_",
    "indoor_churchwindow",
    "indoor_court",
    "indoor_doorwindows",
    "indoor_marketstore",
    "indoor_room",
    "indoor_venue",
    "dark_light",
    "others_",
    "outdoor_",
    "outdoor_city",
    "outdoor_field",
    "outdoor_grass",
    "outdoor_house",
    "outdoor_mountain",
    "outdoor_oceanbeach",
    "outdoor_playground",
    "outdoor_railway",
    "outdoor_road",
    "outdoor_sportsfield",
    "outdoor_stonerock",
    "outdoor_street",
    "outdoor_water",
    "outdoor_waterside",
    "people_",
    "people_baby",
    "people_crowd",
    "people_group",
    "people_hand",
    "people_many",
    "people_portrait",
    "people_show",
    "people_tattoo",
    "people_young",
    "plant_",
    "plant_branch",
    "plant_flower",
    "plant_leaves",
    "plant_tree",
    "object_screen",
    "object_sculpture",
    "sky_cloud",
    "sky_sun",
    "people_swimming",
    "outdoor_pool",
    "text_",
    "text_mag",
    "text_map",
    "text_menu",
    "text_sign",
    "trans_bicycle",
    "trans_bus",
    "trans_car",
    "trans_trainstation",
]

class VisionApi:
    def __init__(self, subscription_key):
        self.subscription_key = subscription_key

    # Analyze image and return the features
    def analyze_image(self, image_file):
        image_data = None

        with open(image_file, 'rb') as f:
            image_data = f.read()
        
        if not image_data:
            print("Error reading file...")
            return

        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-Type': 'application/octet-stream'
        }
        params = {
            'visualFeatures': 'Adult,Categories,Description,Color,Faces,ImageType,Tags'
        }

        features = None
        
        try:
            response = requests.post(vision_analyze_url, headers=headers, params=params, data=image_data)
            response.raise_for_status()
            analysis = response.json()

            # extract features from analysis
            features = self.extract_features(analysis)
        
        except Exception as e:
            print(str(e))
            features = [0, 0, 0, 0, 0, 0, 0, 0]

        

        return features
    
    def extract_features(self, analysis_json):
        features = [
            # categories
            self.extract_categories(analysis_json.get("categories")),

            # tags
            self.extract_tags(analysis_json.get("tags")),

            # description tags
            len(analysis_json.get("description").get("tags")),

            # description captions
            len(analysis_json.get("description").get("captions")),

            # faces
            self.extract_faces(analysis_json.get("faces")),

            # color
            self.extract_color(analysis_json.get("color")),

            # image type (clipart)
            analysis_json.get("imageType").get("clipArtType"),

            # image type (line)
            analysis_json.get("imageType").get("lineDrawingType")
        ]
        return features
    

    def extract_categories(self, categories):
        return len(categories)
    
    def extract_tags(self, tags):
        return len(tags)
    
    def extract_faces(self, faces):
        return len(faces)

    def extract_color(self, color):
        return int(color.get("isBwImg"))

