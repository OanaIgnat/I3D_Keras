import json

def read_time_per_clip():
    with open("data/time_clips.json") as json_file:
        time_clips = json.load(json_file)

    return time_clips

def read_results_i3d():
    with open('data/results.txt', 'r') as file:
        data = file.read().replace('\n', '|')

    dict_results = {}
    list_results_per_video = data.split("For video")[1:]
    for i in range(len(list_results_per_video)):
        clip, norm_logits, probabilities, _ = list_results_per_video[i].split("||")
        clip = clip[1:-2] + ".mp4"
        norm_logits = norm_logits.split("Norm of logits: ")[1]
        probabilities = probabilities[len("Top 20 classes and probabilities "):].split("|")
        probs = []
        for i in probabilities:
            i = ", ".join(i.split(" ")[:2]) + ", " + " ".join(i.split(" ")[2:])
            probs.append(i)
        top_2_prob = probs[:2]
        dict_results[clip] = top_2_prob

    return dict_results

def read_actions_time():
    with open("data/actions_sent_time.json") as json_file:
        actions_sent_time = json.load(json_file)

    for key in actions_sent_time.keys():
        for value in actions_sent_time[key]:
            print(value)


def combine_time_actions(time_clips, dict_results):
    new_dict = {}
    for clip in time_clips:
        new_dict[clip] = [time_clips[clip], dict_results[clip]]

    for key in new_dict.keys():
        print(new_dict[key])

def main():
    # time_clips = read_time_per_clip()
    # dict_results = read_results_i3d()
    # combine_time_actions(time_clips, dict_results)
    read_actions_time()


if __name__ == "__main__":
    main()