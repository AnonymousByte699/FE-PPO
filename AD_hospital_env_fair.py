import torch
import torch.nn.functional as F
import pandas as pd


# AD hospital Env
class ADEnv(object):
    def __init__(self, rate):
        self.rate = rate

        # load data
        hospitals = pd.read_excel('dataset/beijing_hospitals_data.xlsx')
        people = pd.read_excel('dataset/community_elderly_data.xlsx')

        # hospital location
        hospitals_loc_x = torch.tensor(hospitals.loc[:, 'gcj-x'])
        hospitals_loc_y = torch.tensor(hospitals.loc[:, 'gcj-y'])

        # doctors num
        self.doctor_num = torch.tensor(hospitals.loc[:, '总数量'])
        # time rate
        detect_rate = 5.5 / 100
        self.patients_avail = self.doctor_num * 2860 * detect_rate

        # drug
        bj_usa = 21893095 * 19.6 / 100 * 19 / 100 / 14900000
        self.drug_supply = (2.7 + 4) / 2 * 1000000 * bj_usa / 2.3
        self.drug_num = torch.zeros(self.doctor_num.shape[0])

        # community location
        olds_loc_x = torch.tensor(people.loc[:, '经度'])
        olds_loc_y = torch.tensor(people.loc[:, '纬度'])

        # distance (339 * 29)
        distance_x = (olds_loc_x.reshape(olds_loc_x.shape[0], -1) - hospitals_loc_x.unsqueeze(0)) ** 2
        distance_y = (olds_loc_y.reshape(olds_loc_y.shape[0], -1) - hospitals_loc_y.unsqueeze(0)) ** 2
        distance = torch.sqrt(distance_x + distance_y)
        self.region_distance = (distance - torch.min(distance, dim=0).values) / (torch.max(distance, dim=0).values - torch.min(distance, dim=0).values).unsqueeze(0)
        distance = distance - torch.mean(distance, dim=1).unsqueeze(1)
        self.distance = F.normalize(distance, p=2, dim=1)

        # elderly num
        self.olds_num = torch.tensor(people.loc[:, '2020常住人口'] * people.loc[:, '2020年龄段比例60以上'] / 100)

        # gender rate
        self.olds_gender = torch.tensor(people.loc[:, '2020男性/女性'])

        # education rate
        olds_edu = torch.tensor(people.loc[:, '2020平均受教育年限（15岁以上）'])
        self.olds_edu = (olds_edu - torch.mean(olds_edu, dim=0)) / torch.std(olds_edu)

        # states
        self.state_trans = torch.tensor([
            [0, 3 / 100, 0, 0],
            [0, 0, 0, 6.5 / 100],
            [0, 0, 0, 0.1 / 100],
            [0, 0, 0, 0]
        ])
        self.state = torch.zeros((self.olds_num.shape[0], self.state_trans.shape[0]))

        # AD rate
        ad_rate_60 = 3.9 / 100
        self.state[:, 0] += (self.olds_num * (1 - ad_rate_60))
        self.state[:, 1] += self.olds_num * ad_rate_60

    # willing rate
    def _willing_rate(self, drug_num):
        # distance
        willing = -0.8 * self.distance
        # drug num and doctor num
        if torch.std(drug_num, dim=0) <= 0.0005:
            drug_num = torch.zeros(drug_num.shape)
        else:
            drug_num = (drug_num - torch.mean(drug_num, dim=0)) / torch.std(drug_num, dim=0)
        doctor_num = (self.doctor_num - torch.mean(self.doctor_num, dim=0)) / torch.std(self.doctor_num, dim=0)
        willing += 1.3 * drug_num.reshape(1, -1) + 0.3 * doctor_num.reshape(1, -1)
        willing = F.softmax(willing, dim=1)

        # mci rate
        mci_rate_60 = 19 / 100
        base = torch.ones(self.olds_num.shape[0])
        base *= ((32.14 - 14) / 100 / 12 * (1 - mci_rate_60) + 32.14 / 100 / 12 * mci_rate_60)
        # education rate
        base += 0.01 * self.olds_edu
        # gender rate
        base *= (self.olds_gender + 0.96 * 100) / (100 + self.olds_gender)
        willing *= base.unsqueeze(1)

        willing = torch.clamp(willing, min=0.0)
        return willing

    def step(self, distribution):
        distribution = F.softmax(torch.tensor(distribution), dim=0)
        self.drug_num = self.drug_supply * distribution

        worsen = self.get_worsen()
        olds_avail = self.iteration()

        reward = torch.sum(olds_avail) / torch.sum(self.olds_num) * 1400
        reward /= torch.exp(110 * (torch.max(self.get_region_avail()) - torch.min(self.get_region_avail())))
        return reward

    def step_none(self, distribution):  # baseline without softmax
        # print(distribution)
        self.drug_num = self.drug_num + self.drug_supply * distribution

        worsen = self.get_worsen()
        olds_avail = self.iteration()

        reward = torch.sum(olds_avail) / torch.sum(self.olds_num)
        return reward  # reward of olds avail ratio

    def iteration(self):
        willing_rate = self._willing_rate(self.drug_num)
        avail = self.drug_num

        self.olds_willing = (self.state[:, 0] + self.state[:, 1]).unsqueeze(1) * willing_rate

        olds_avail = torch.min(torch.sum(self.olds_willing, dim=0), avail)

        # print("drug: ", self.drug_num)
        # print("avail: ", olds_avail, torch.sum(olds_avail))
        self.drug_num = self.drug_num - olds_avail

        hospital_olds_rate = self.olds_willing / torch.sum(self.olds_willing, dim=0).unsqueeze(0)
        olds_to_hospital = olds_avail.unsqueeze(0) * hospital_olds_rate
        self.olds_region_avail = torch.sum(olds_to_hospital, dim=1)
        olds_avail_state_1 = self.olds_region_avail * (self.state[:, 1] / (self.state[:, 0] + self.state[:, 1]))

        # yes to treating
        self.state[:, 1] = self.state[:, 1] - olds_avail_state_1
        self.state[:, 2] = self.state[:, 2] + olds_avail_state_1

        # yes / treating to worsen
        yes_to_worsen = self.state[:, 1] * self.state_trans[1, 3]
        treating_to_worsen = self.state[:, 2] * self.state_trans[2, 3]
        self.state[:, 3] = self.state[:, 3] + yes_to_worsen + treating_to_worsen
        self.state[:, 1] = self.state[:, 1] - yes_to_worsen
        self.state[:, 2] = self.state[:, 2] - treating_to_worsen

        # no to yes
        no_to_yes = self.state[:, 0] * self.state_trans[0, 1]
        self.state[:, 1] = self.state[:, 1] + no_to_yes
        self.state[:, 0] = self.state[:, 0] - no_to_yes

        return torch.sum(olds_avail)

    def get_worsen(self):
        return torch.sum(self.state[:, 3], dim=0)

    def get_region_avail(self):
        return self.olds_region_avail / torch.sum(self.olds_willing, dim=1)

    def get_obs(self):
        # return torch.cat((self.drug_num, self.patients_avail)).reshape((1, -1)).squeeze(0)
        return torch.cat((
            torch.mm(self.region_distance.T, self.state.to(torch.double)).reshape((1, -1)).squeeze(0),
            self.patients_avail)).squeeze(0)

    def reset(self):
        self.state = torch.zeros((self.olds_num.shape[0], self.state_trans.shape[0]))
        ad_rate_60 = 3.9 / 100
        self.state[:, 0] += (self.olds_num * (1 - ad_rate_60))
        self.state[:, 1] += self.olds_num * ad_rate_60
        self.drug_num = torch.zeros(self.doctor_num.shape[0])

