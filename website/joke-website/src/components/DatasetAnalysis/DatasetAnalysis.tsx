import React from 'react';
import {
  VictoryChart,
  VictoryBar,
  VictoryAxis
} from "victory"

const DatasetAnalysis: React.FC = () => {
  return (
    <div className="container dataset-analysis">
      <div className="container-header">
        <h2>Dataset Analysis</h2>
      </div>

      There were three datasets, named Reddit jokes dataset, Wocka jokes dataset, and the StupidStuff jokes dataset.
      The size of the datasets is shown in the graphs below.
      {/* All three datasets can be found at <link href='https://github.com/taivop/joke-dataset'>here</link>. */}
      <div className='number-jokes-chart-container'>
        <div className='victory-bar-chart'>
          <VictoryChart
            height={ 200 }
            padding={ { left: 100, top: 60, bottom: 60, right: 100 } }
            domainPadding={ 10 }>
            <VictoryBar horizontal
              barWidth={ 20 }
              style={ { data: { fill: "#dd3388" } } }
              data={ [
                { x: 'reddit', y: 195000 },
                { x: 'wocka', y: 10000 },
                { x: 'stupidjokes', y: 3770 }
              ] }
            />
            <VictoryAxis padding={ 200 } />
            <VictoryAxis dependentAxis
              tickValues={[0, 50000, 100000, 150000, 200000]}
              label='Number of jokes' />
          </VictoryChart>
        </div>
      </div>
      <h3>Reddit</h3>
      <p>
      This dataset consists of jokes parsed from the r/jokes subreddit on reddit.com.
      The dataset can be found here(MAKE THIS A LINK).
      </p>
      <p>
      The reddit dataset has the schema { '{ id, score, title, body }' }.
      </p>
      <p>
      While this dataset has the largest number of jokes by far, it quickly became clear that this dataset required the most attention in terms of data processing.
      Many of the jokes in the dataset contained many non ascii characters, which greatly increased the output space and proved to be a significant hinderance in producing coherent English for the jokes.
      </p>
      <h3>Wocka</h3>
      <h3>StupidStuff</h3>
    </div>
  )
}

export default DatasetAnalysis