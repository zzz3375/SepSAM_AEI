import { merge } from "webpack-merge";
import { resolve } from "path";
import Dotenv from "dotenv-webpack";
import commonConfig from "./common";

const prodConfig = merge(commonConfig, {
  mode: "production",
  output: {
    filename: "js/bundle.[contenthash].min.js",
    path: resolve(__dirname, "../../dist"),
    publicPath: "/",
  },
  devtool: "source-map",
  plugins: [new Dotenv()],
});

export default prodConfig;
