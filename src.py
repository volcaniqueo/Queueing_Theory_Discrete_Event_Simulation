import heapq
import random
import numpy as np
import math
#import pandas as pd
import statistics as stat
random.seed(2019400033 + 2020400078) #Seeds are calculated with addition of student ID's.
np.random.seed(2019400033 + 2020400078)

#Class representing event in the simulation model.
class Event:
    #Constructor method
    def __init__(self, time, event_type, patient, medical_service, duration):
        self.time = time                        #Event's time
        self.event_type = event_type            #Event's type (Arrival | Departure_Triage | Treated_at_Home | Treated_at_Hospital)  
        self.patient = patient                  #Event's patient  
        self.medical_service = medical_service  #Event's service type (Nurse Object | Bed Object | None)
        self.duration = duration                #Event's duration, calculated with functions (generate_interarrival() etc.)
    
    #Comparator for event class used in heapq
    def __lt__(self, other):
        return self.time < other.time 
    #ToString method for event class 
    def __str__(self):
        details = "[" + self.event_type + ", Patient:" + str(self.patient.id) + "]"
        return details
 
#Class representing patient in the simulation model.
class Patient:
    def __init__(self, id):
        self.id = id            #ID of the patient.
        self.enter_time = 0     #Time patient enters the system with Arrival event.
        self.exit_time = 0      #Time patient exits the system with  Treated_at_Home or Treated_at_Hospital.

#Class representing triage nurse in the simulation model.
class Nurse:
    def __init__(self, id):
        self.id = id            #ID of the nurse.
        self.worked_time = 0    #Total utilization time of the nurse. 

#Class representing hospital bed in the simulation model.
class Bed:
    def __init__(self, id):
        self.id = id            #ID of the bed.
        self.occupied_time = 0  #Total utilization time of the bed. 
    
#Class representing the hospital in the simulation model.
class HealthcareSystem:
    def __init__(self, S, K, mu_t, mu_cb, mu_s, myLambda, p1, healed_patients_limit, start_type):
        self.healed_patients = 0                            #Total number of healed patients.
        self.Lh = 0                                         #Current number of patients at the home treatment. 
        self.Xs = 0                                         #Random number to compare with p1
        self.time = 0                                       #Current system time
        self.patient_id = 0                                 #Variable to assign unique ids to patient coming with Arrival event.
        self.healed_patients_limit = healed_patients_limit  #Limiting condition depends on healed_patients
        self.S = S                                          #Total number of triage nurses.
        self.K = K                                          #Total number of hospital beds.
        self.mu_t = mu_t            #Parameter to generate triage service time. (exponential)
        self.mu_cb = mu_cb          #Parameter to generate hospital bed service time. (exponential) 
        self.mu_s = mu_s            #Parameter to generate home treatment time for stable condition. (exponential) 
        self.myLambda = myLambda    #Parameter to generate interarrival time. (exponential)
        self.p1 = p1                #Decision parameter for patient's condition.(stable or critical)
        self.patient_queue = []     #Array to store patients waiting for triage
        self.event_queue = []       #Priority queue for events.(heapq is used)
        self.num_patients_arrived = 0           #Total number of patients arrived with Arrival event.
        self.num_patients_directly_triage = 0   #Total number of patients went to triage without waiting.
        self.num_patients_waiting_triage = 0    #Total number of patients waited for triage.
        self.num_patients_arrived_beds = 0      #Total number of patients in critical condition.
        self.num_patients_directly_beds = 0     #Total number of critical patients treated at beds in the hospital.
        self.num_patients_rejected_beds = 0     #Total number of critical patients treated at home.
        self.time_triage_empty = 0              #Total time all nurses are idle.
        self.start_time_for_empty = 0           #Set to system's current time whenever all nurses are idle.
        self.time_beds_empty = 0                #Total time all beds are idle.
        self.start_time_for_empty_beds = 0      #Set to system's current time whenever all beds are idle.
        self.start_type = start_type            #Start condition of the system (full or empty or half) for both triage and beds.
        self.time_array = []                    #Used in logger function for creating table with pandas
        self.type_array = []                    #Used in logger function for creating table with pandas
        self.system_number_array = []           #Used in logger function for creating table with pandas
        self.triage_number_array = []           #Used in logger function for creating table with pandas
        self.beds_number_array = []             #Used in logger function for creating table with pandas
        self.queue_number_array = []            #Used in logger function for creating table with pandas
        self.healed_number_array = []           #Used in logger function for creating table with pandas
        self.interarrival_array = []            #Stores interarrival times of the patients.
        self.nurse_service_array = []           #Stores service times of the nurses.
        self.hospital_healing_array = []        #Stores healing times of the critical patients in hospital. 
        self.home_healing_array_s = []          #Stores healing times of the stable patients in home.
        self.home_healing_array_c = []          #Stores healing times of the critical patients in home.
        self.patient_id_array = []              #Used in logger function for creating table with pandas.
        self.fel_array = []                     #Used in logger function for creating table with pandas.
        self.log_count = 0                      #Used to limit the logger function to print desired number of total events.
        self.available_nurse_list = []          #List of currently idle nurses.(FIFO)
        self.available_bed_list = []            #List of currently idle beds.(FIFO)
        self.bed_list = []                      #List of all Bed object instances in the system.
        self.nurse_list = []                    #List of all Nurse object instances in the system.
        self.patient_list = []                  #List of all Patient object instances in the system.
        self.treated_home = 0                   #Total number of patients treated at home.
        self.treated_hospital = 0               #Total number of patients treated at hospital.
        self.start_time_for_full_beds = 0       #Set to system's current time whenever all beds are busy.
        self.start_time_for_full_triage = 0     #Set to system's current time whenever all nurses are busy.
        self.time_beds_full = 0                 #Total time all beds are busy.
        self.time_triage_full = 0               #Total time all nurses are busy.

    #Function to generate exponential interarrival times with parameter myLambda.
    def generate_interarrival(self):
        scale = 1 / self.myLambda
        value = float(np.random.exponential(scale=scale, size=1))
        self.interarrival_array.append(value)
        return value

    #Function to generate exponential service time for triage nurses with parameter mu_t.
    def generate_nurse_service_time(self):
        scale = 1 / self.mu_t
        value = float(np.random.exponential(scale=scale, size=1))
        self.nurse_service_array.append(value)
        return value

    #Function to generate exponential service time for hospital beds with parameter mu_cb.
    def generate_hospital_healing_time(self):
        scale = 1 / self.mu_cb
        value = float(np.random.exponential(scale=scale, size=1))
        self.hospital_healing_array.append(value)
        return value

    #Function to generate exponential home healing times. heal_type = "s" for stable, "c" for critical patients.
    def generate_home_healing_time(self, heal_type):
        if heal_type == "s":
            scale = 1 / self.mu_s
            value = float(np.random.exponential(scale=scale, size=1))
            self.home_healing_array_s.append(value)
            return value
        else:
            alpha = ((1.75 - 1.25) * random.random()) + 1.25
            scale =  1 / (self.mu_cb / alpha)
            #normal_time = self.generate_hospital_healing_time()
            #value = alfa * normal_time
            value = float(np.random.exponential(scale=scale, size=1))
            self.home_healing_array_c.append(value)
            return value

    #Function representing arrival of a new patient to the system. 
    #If there is an idle nurse in the triage, patient is directed to triage which results in creation necessary Departure_Triage event,
    #that is added to the event_queue. 
    #If all nurses are busy, patient is directed to the waiting queue.
    #New Arrival event is created after arrival of a patient.
    #All necessary state variables are updated. (This is self-explanatory in code.)
    def arrival(self, event):
        self.Lsys += 1                                                 
        patient = event.patient
        self.num_patients_arrived += 1
        event.patient.enter_time = event.time
        self.patient_list.append(event.patient)
        if self.Lt < self.S:
            if (self.empty_check):
                self.time_triage_empty += (self.time - self.start_time_for_empty)
                self.empty_check = False
            nurse = self.available_nurse_list.pop(0)
            if (len(self.available_nurse_list) == 0):
                self.start_time_for_full_triage = self.time

            self.Lt += 1
            self.num_patients_directly_triage += 1
            random_duration = self.generate_nurse_service_time()
            event = Event(time=(self.time + random_duration), event_type="Departure_Triage",  patient=patient, medical_service=nurse, duration=random_duration)
            heapq.heappush(self.event_queue, event)
        else:
            self.Lq += 1
            self.num_patients_waiting_triage += 1
            self.patient_queue.append(patient)
        patient = Patient(self.patient_id)
        self.patient_id += 1
        event = Event(time=(self.time + self.generate_interarrival()), event_type="Arrival", patient=patient, medical_service=None, duration=0)
        heapq.heappush(self.event_queue, event)

    #Function representing departure of a patient from triage.
    #Random number Xs is used to determine patient's condition.
    #If patient is found to be in critical condition and there is an idle bed in the hospital, patient is directed to the bed.
    #If patient is found to be in critical condition and there isn't any idle bed in the hospital, patient is directed to the home.
    #If patient is found to be in stable condition, patient is directed to the home. 
    #Conditions above triggers creation of Treated_at_Hospital, Treated_at_Home, Treated_at_Home events respectively.
    #   these events are added to the event_queue.
    #After these if there is a patient in the triage queue, directed to triage which results in creation necessary Departure_Triage event,
    #   that is added to the event_queue.
    #All necessary state variables are updated. (This is self-explanatory in code.)
    def departure_triage(self, event):
        patient = event.patient
        self.Lt -= 1
        self.available_nurse_list.append(event.medical_service)
        if (len(self.available_nurse_list) == 1):
            self.time_triage_full += (self.time - self.start_time_for_full_triage)
        event.medical_service.worked_time += event.duration
        if (self.Lt == 0):
            self.start_time_for_empty = self.time
            self.empty_check = True
        self.Xs = random.random()
        if self.Xs < self.p1:
            self.Lh += 1
            random_duration = self.generate_home_healing_time("s")
            event = Event(time=(self.time + random_duration), event_type="Treated_at_Home",  patient=patient, medical_service=None, duration=random_duration)
            heapq.heappush(self.event_queue, event)
        else:
            self.num_patients_arrived_beds += 1
            if self.Lb < self.K:
                if(self.beds_empty_check):
                    self.time_beds_empty += (self.time - self.start_time_for_empty_beds)
                    self.beds_empty_check = False
                self.num_patients_directly_beds += 1
                self.Lb += 1
                bed = self.available_bed_list.pop(0)
                if (len(self.available_bed_list) == 0):
                    self.start_time_for_full_beds = self.time
                random_duration = self.generate_hospital_healing_time()
                event = Event(time=(self.time + random_duration), event_type="Treated_at_Hospital",  patient=patient, medical_service=bed, duration=random_duration)
                heapq.heappush(self.event_queue, event)
            else:
                self.num_patients_rejected_beds += 1
                self.Lh += 1
                random_duration = self.generate_home_healing_time("c")
                event = Event(time=(self.time + random_duration), event_type="Treated_at_Home",  patient=patient, medical_service=None, duration=random_duration)
                heapq.heappush(self.event_queue, event)

        if self.Lq > 0:
            patient = self.patient_queue.pop(0)
            self.Lq -= 1
            if (self.empty_check):
                self.time_triage_empty += (self.time - self.start_time_for_empty)
                self.empty_check = False
            self.Lt += 1
            nurse = self.available_nurse_list.pop(0)
            if (len(self.available_nurse_list) == 0):
               self.start_time_for_full_triage = self.time
            random_duration = self.generate_nurse_service_time()
            event = Event(time=(self.time + random_duration), event_type="Departure_Triage",  patient=patient, medical_service=nurse, duration=random_duration)
            heapq.heappush(self.event_queue, event)
    
    #Function to make necessary state updates whenever a patient is treated at home.
    def treated_at_home(self, event):
        self.Lh -= 1
        self.healed_patients += 1
        self.Lsys -= 1
        event.patient.exit_time = event.time
        self.treated_home += 1

    #Function to make necessary state updates whenever a patient is treated at hospital.
    def treated_at_hospital(self, event):
        self.Lb -= 1
        self.healed_patients += 1
        self.Lsys -= 1
        event.patient.exit_time = event.time
        if(self.Lb == 0):
            self.start_time_for_empty_beds = self.time
            self.beds_empty_check = True
        self.available_bed_list.append(event.medical_service)
        if (len(self.available_bed_list) == 1):
            self.time_beds_full += (self.time - self.start_time_for_full_beds)
        event.medical_service.occupied_time += event.duration
        self.treated_hospital += 1
    
    #Function used to advance system time according to the event.
    def advance_time(self, event):
        self.time = event.time

    #General function to execute different events according to their types. Also logger function is triggered to store the history limited with log_count. 
    def execute_event(self, event):
        self.advance_time(event)
        event_type = event.event_type
        if event_type == "Arrival":
            self.arrival(event)            
        elif event_type == "Departure_Triage":
            self.departure_triage(event)
        elif event_type == "Treated_at_Home":
            self.treated_at_home(event)
        elif event_type == "Treated_at_Hospital":
            self.treated_at_hospital(event)
        if self.log_count < 50:
            self.logger(event)
            self.log_count += 1

    #Function to initialize the simulation with necessary values and creations of objects.
    def initialize_simulation(self):
        for i in range(self.S):
            nurse = Nurse(i)
            self.available_nurse_list.append(nurse)
            self.nurse_list.append(nurse)
        for i in range(self.K):
            bed = Bed(i)
            self.available_bed_list.append(bed)
            self.bed_list.append(bed)
        
        if self.start_type == "empty":
            self.Lq = 0
            self.Lt = 0
            self.Lb = 0
            self.Lsys = 0
            self.empty_check = True
            self.beds_empty_check = True
        else:
            self.empty_check = False
            self.beds_empty_check = False
            if (self.start_type == "half"):
                num_triage = math.floor(self.S / 2)
                num_bed = math.floor(self.K / 2)
            else:
                num_triage = self.S
                num_bed = self.K
            self.Lq = 0
            self.Lt = num_triage
            self.Lb = num_bed
            self.Lsys = num_triage + num_bed
            for index in range(num_triage):
                patient = Patient(self.patient_id)
                self.patient_id += 1
                nurse = self.available_nurse_list.pop(0)
                random_duration = self.generate_nurse_service_time()
                event = Event(time=(self.time + random_duration), event_type="Departure_Triage",  patient=patient, medical_service=nurse, duration=random_duration)
                heapq.heappush(self.event_queue, event)
            if (len(self.available_nurse_list) == 0):
               self.start_time_for_full_triage = self.time
            for index in range(num_bed):
                patient = Patient(self.patient_id)
                self.patient_id += 1
                bed = self.available_bed_list.pop(0)
                random_duration = self.generate_hospital_healing_time()
                event = Event(time=(self.time + random_duration), event_type="Treated_at_Hospital",  patient=patient, medical_service=bed, duration=random_duration)
                heapq.heappush(self.event_queue, event)
            if (len(self.available_bed_list) == 0):
                self.start_time_for_full_beds = self.time
            

        patient = Patient(self.patient_id)
        self.patient_id += 1
        event = Event(time=self.time, event_type="Arrival", patient=patient, medical_service=None, duration=0)
        self.execute_event(event)

    #Function used to run the simulation.
    def run_simulation(self):
        
        self.initialize_simulation()      

        while self.healed_patients < self.healed_patients_limit:
            event = heapq.heappop(self.event_queue)
            self.execute_event(event)        

    #Function used to store the simulation data. Used for pandas  
    def logger(self,event):
        self.time_array.append(float(event.time))
        self.type_array.append(event.event_type)
        self.system_number_array.append(self.Lsys)
        self.triage_number_array.append(self.Lt)
        self.beds_number_array.append(self.Lb)
        self.queue_number_array.append(self.Lq)
        self.healed_number_array.append(self.healed_patients)
        self.patient_id_array.append(event.patient.id)
        temp = []
        for i in self.event_queue:
            if (i.event_type == "Arrival"):
                details = "[" + "A" + " @ " + str(i.time) + " P:" + str(i.patient.id) + "]"
            elif (i.event_type == "Departure_Triage"):
                details = "[" + "DT" + " @ " + str(i.time) + " P:" + str(i.patient.id) + "]"
            elif (i.event_type == "Treated_at_Home"):
                details = "[" + "T_Hm" + " @ " + str(i.time) + " P:" + str(i.patient.id) + "]"
            elif (i.event_type == "Treated_at_Hospital"):
                details = "[" + "T_H" + " @ " + str(i.time) + " P:" + str(i.patient.id) + "]"
            temp.append(details)
        self.fel_array.append(temp)
        
        

if __name__ == "__main__":
    S = 4
    mu_t = 0.357142857
    p1 = 0.2
    K = 7
    mu_cb = 0.142857143
    mu_s = 0.16
    myLambda = 1
    healed_patients_limit = 20
    start_type = "empty"
    
    system = HealthcareSystem(S, K, mu_t, mu_cb, mu_s, myLambda, p1, healed_patients_limit, start_type)
    system.run_simulation()
    
    #50 EVENTS
    #data_dict = {"Time" : system.time_array, "Event Type" : system.type_array, "Patient ID": system.patient_id_array, "L" : system.system_number_array, "Lt" : system.triage_number_array ,
    #           "Lb:" : system.beds_number_array, "Lq" : system.queue_number_array , "Healed" : system.healed_number_array}
    #df = pd.DataFrame(data_dict)
    """
    print(system.interarrival_array)
    print(system.home_healing_array_s)
    print(system.home_healing_array_c)
    print(system.nurse_service_array)
    print(system.hospital_healing_array)
    """
    # Long-run being empty triage:
    
    print("Long-run probability of arriving patient finds an available nurse")
    print((system.time - system.time_triage_full) / system.time)
    print("--")
    
    # Long-run being empty bed:
    print("Long-run probability of critical patient finds an available bed")
    print((system.time - system.time_beds_full) / system.time)
    print("--")
    
    # Joint:
    
    print("Joint")
    print(((system.time - system.time_beds_full) / system.time) * ((system.time - system.time_triage_full) / system.time))
    print("--")
    
    # Average num people rejected due bed unavailability:
    
    print("Average number of people rejected due to bed unavailability")
    print(system.num_patients_rejected_beds / system.num_patients_arrived_beds) 
    print("--")
    
    # Average utilization of each nurse:
    
    print("Average utilization of each nurse")
    utillist1 = []
    for i in system.nurse_list:
        worked = i.worked_time
        util = (worked / system.time)
        utillist1.append(util)
    print(stat.mean(utillist1))
    print("--")
    
    # Average number of occupied beds:
    
    print("Average number of occupied beds")
    utillist2 = []
    for i in system.bed_list:
        occupied = i.occupied_time
        util = (occupied / system.time)
        utillist2.append(util)
    avg = stat.mean(utillist2)
    print(avg)
    print("--")
    
    # Average number of patients treated at home:
    
    print("Average number of patients treated at home")
    print(system.treated_home / (system.num_patients_arrived))
    
    # Average time a sick person gets better:
    
    print("Average time a sick person gets better")
    time_spent_list = []
    for i in system.patient_list:
        added_time = i.exit_time - i.enter_time
        if added_time > 0:
            time_spent_list.append(added_time)
    print(stat.mean(time_spent_list))


