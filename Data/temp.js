const fs = require("fs");

var data = JSON.parse(fs.readFileSync("iris.json", 'UTF-8'));

data.forEach(ele => {
  temp = ele["class"]
  if (temp == "setosa") {
    ele["class"] = 1
  } else if (temp == "virginica") {
    ele["class"] = 2
  } else {
    ele["class"] = 3
  }
});

fs.writeFileSync("iris_new.json", JSON.stringify(data), 'UTF-8')