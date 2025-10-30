import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "Airplane": "airplane",
    "Airport": "airport",
    "Bareland": "bareland",
    "BaseballField": "baseball field",
    "BasketballCourt": "basketball court",
    "Beach": "beach",
    "Bridge": "bridge",
    "Buildings": "buildings",
    "Cemetry": "cemetry",
    "Center": "center",
    "Chaparral": "chaparral",
    "ChristmasTreeFarm": "christmas tree farm",
    "Church": "church",
    "ClosedRoad": "closed road",
    "Cloud": "cloud",
    "CoastalMansion": "coastal mansion",
    "Commercial": "commercial",
    "DenseResidential": "dense residential",
    "Desert": "desert",
    "ErodedFarmland": "eroded farmland",
    "Farmland": "farmland",
    "FootballField": "football field",
    "Forest": "forest",
    "Freeway": "freeway",
    "GolfCourse": "golf course",
    "Industrial": "industrial",
    "Intersection": "intersection",
    "Island": "island",
    "Lake": "lake",
    "Meadow": "meadow",
    "MediumResidential": "medium residential",
    "MobileHomePark": "mobile home park",
    "Mountain": "mountain",
    "NursingHome": "nursing home",
    "OilGasField": "oil gas field",
    "OilWell": "oil well",
    "Palace": "palace",
    "Park": "park",
    "ParkingLot": "parking lot",
    "Parkway": "parkway",
    "Playground": "playground",
    "Port": "port",
    "Railway": "railway",
    "RailwayStation": "railway station",
    "Resort": "resort",
    "River": "river",
    "Roundabout": "roundabout",
    "Runway": "runway",
    "RunwayMarking": "runway marking",
    "School": "school",
    "SeaIce": "sea ice",
    "Ship": "ship",
    "ShippingYard": "shipping yard",
    "Snowberg": "snow berg",
    "SolarPanel": "solar panel",
    "SparseResidential": "sparse residential",
    "Square": "square",
    "Stadium": "stadium",
    "StorageTank": "storage tank",
    "SwimmingPool": "swimming pool",
    "TennisCourt": "tennis court",
    "Terrace": "terrace",
    "ThermalPowerStation": "thermal power station",
    "TransmissionTower": "transmission tower",
    "VegetableGreenhouse": "vegetable greenhouse",
    "Viaduct": "viaduct",
    "WastewaterTreatmentPlant": "wastewater treatment plant",
    "Wetland": "wetland",
    "WindTurbine": "wind turbine"
}


@DATASET_REGISTRY.register()
class RSICD(DatasetBase):

    dataset_dir = "RSICD"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "annotations.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
