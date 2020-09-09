TODOPath = '.\';
TODOFile = 'FILL THIS HERE!';
jsonData = jsondecode(fileread(strcat(TODOPath, TODOFile, ".json")));

dataset_joints = {'rank','rkne','rhip','lhip','lkne','lank','pelv','thor','neck','head','rwri','relb','rsho','lsho','lelb','lwri'};

jnt_missing = 1 - [jsonData.joints_vis];

new_pos = [jsonData.joints];
new_pos(new_pos == -1) = 0;
pos_gt_src = reshape(new_pos,16,2,[]);

headboxes_src = reshape([jsonData.bbox],2,2,[]);

save(strcat(TODOFile,'.mat'),'dataset_joints','headboxes_src','jnt_missing','pos_gt_src');
